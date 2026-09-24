"""전처리 파이프라인 본체.

역할: raw 로드부터 정제, 분할, 검증, 저장까지의 실행 순서를 여기서만 정한다
    (단계를 빼거나 바꾸는 ablation 실험을 한 곳에서 하기 위함).
동작:
    1. load_raw로 raw CSV를 읽는다.
    2. articles -> customers -> transactions 순서로 정제한다
       (거래 정제에 남길 상품 목록이 필요하므로 articles가 먼저다).
    3. week_idx로 train / valid / test를 나눈다.
    4. 단계별 개수를 logging과 report dict에 기록하고 validate로 검증한다.
    5. 검증을 통과하면 parquet 5개와 preprocess_report.json을 저장한다.
    config의 transactions_nrows가 설정되면 dry run이다. 빈 분할은 경고로만 남기고
    아무것도 저장하지 않는다.
"""

import json
import logging
import os
import pandas as pd
from src.data import articles as art
from src.data import customers as cus
from src.data import transactions as trx
from src.data.load import load_raw
from src.data.split import split_by_week

KEY_COLUMNS = {
    "transactions": ["article_id", "customer_id", "t_dat"],
    "articles": ["article_id", "product_type_name", "garment_group_name"],
    "customers": ["customer_id", "age_group"],
}

logger = logging.getLogger(__name__)


def _record(report: dict, step: str, before: int, after: int) -> None:
    """단계 전후 개수를 로그로 남기고 report에 누적한다.

    Args:
        report: 누적할 리포트 dict (steps 키에 추가).
        step: 단계 이름.
        before: 단계 전 개수.
        after: 단계 후 개수.
    """
    removed = before - after
    ratio = removed / before if before else 0.0
    logger.info(
        "[%s] %d -> %d (제거 %d, %.2f%%)", step, before, after, removed, ratio * 100
    )
    report.setdefault("steps", []).append(
        {"step": step, "before": before, "after": after, "removed_ratio": ratio}
    )


def clean_articles(df: pd.DataFrame, config: dict, report: dict) -> pd.DataFrame:
    """spec 6.3 순서대로 articles를 정제한다.

    Args:
        df: raw articles DataFrame.
        config: preprocess config dict.
        report: 누적할 리포트 dict.

    Returns:
        정제된 articles DataFrame.

    Raises:
        ValueError: garment_group_mapping 값이 잘못됐을 때.
    """
    cfg = config["articles"]
    steps = [
        (
            "articles: 제외 product_group",
            lambda d: art.remove_excluded_groups(d, cfg["excluded_product_groups"]),
        ),
        (
            "articles: 비의류 type",
            lambda d: art.remove_non_fashion_types(d, cfg["non_fashion_types"]),
        ),
        (
            "articles: 이미지 없음",
            lambda d: art.filter_articles_with_image(d, config["paths"]["image_dir"]),
        ),
    ]
    for name, step in steps:
        before = len(df)
        df = step(df)
        _record(report, name, before, len(df))

    df = art.merge_product_types(df, cfg["type_merge_mapping"])
    # garment 매핑은 희귀 type 병합 전에 해야 매핑 key가 남아 있다
    df = art.fill_unknown_garment_group(df, cfg["garment_group_mapping"])
    rare = art.find_rare_types(df, cfg["min_type_count"])
    report["rare_types_merged"] = {k: int(v) for k, v in rare.items()}
    df = art.merge_rare_types(df, cfg["min_type_count"], cfg["rare_type_prefix"])
    report["garment_unknown_remaining"] = int(
        (df["garment_group_name"] == "Unknown").sum()
    )
    df = art.fill_detail_desc(df)
    df = art.drop_product_type_no(df)
    logger.info(
        "articles: product_type %d종, garment Unknown %d개 남음",
        df["product_type_name"].nunique(),
        report["garment_unknown_remaining"],
    )
    return df


def clean_customers(df: pd.DataFrame, config: dict, report: dict) -> pd.DataFrame:
    """spec 6.4 순서대로 customers를 정제한다. 유저는 삭제하지 않는다.

    Args:
        df: raw customers DataFrame.
        config: preprocess config dict.
        report: 누적할 리포트 dict.

    Returns:
        정제된 customers DataFrame.

    Raises:
        ValueError: 행 수가 입력과 달라졌을 때.
    """
    cfg = config["customers"]
    n_rows = len(df)
    report["age_missing_raw"] = int(df["age"].isna().sum() + (df["age"] == -1).sum())
    logger.info("customers: 원본 age 결측 %d", report["age_missing_raw"])

    df = cus.add_fashion_news_subscribed(df, cfg["subscribed_frequencies"])
    df = cus.fill_active(df)
    df = cus.fill_club_member_status(df, cfg["unknown_label"])
    df = cus.add_age_group(df, cfg["age_bins"], cfg["age_labels"], cfg["unknown_label"])
    df = cus.drop_customer_columns(df, cfg["drop_columns"])

    if len(df) != n_rows:
        raise ValueError(f"customers 행 수 변경: {n_rows} -> {len(df)}")
    _record(report, "customers", n_rows, len(df))
    logger.info(
        "customers: 구독 %d, age_group 분포 %s",
        df["fashion_news_subscribed"].sum(),
        df["age_group"].value_counts().to_dict(),
    )
    return df


def clean_transactions(
    df: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame, report: dict
) -> pd.DataFrame:
    """spec 6.5 순서대로 transactions를 정제한다.

    Args:
        df: raw transactions DataFrame.
        articles: 정제된 articles (valid_article_ids 출처).
        customers: 정제된 customers.
        report: 누적할 리포트 dict.

    Returns:
        유효 상품 거래만 남고 week_idx가 추가된 DataFrame.
    """
    # week_idx 기준일은 필터 전 원본 거래의 최대 날짜
    max_date = df["t_dat"].max()
    before = len(df)
    df = trx.filter_valid_articles(df, articles["article_id"])
    _record(report, "transactions: 유효 상품", before, len(df))

    n_unknown = trx.count_unknown_customers(df, customers["customer_id"])
    report["unknown_customers_in_transactions"] = n_unknown
    logger.info("transactions: customers에 없는 customer_id %d명", n_unknown)

    df = trx.add_week_idx(df, max_date)
    report["max_date"] = str(max_date.date())
    return df


def validate(
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    n_customers_raw: int,
    transactions: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    dry_run: bool,
) -> None:
    """spec 6.7 검증. dry run에서는 빈 분할로 인한 실패를 경고로 바꾼다.

    Args:
        articles: 정제된 articles.
        customers: 정제된 customers.
        n_customers_raw: 원본 customers 행 수.
        transactions: 정제된 transactions (분할 전).
        splits: {"train", "valid", "test"} -> DataFrame.
        dry_run: nrows 샘플 실행 여부.

    Raises:
        ValueError: 검증 실패 시.
    """
    errors = []
    if not transactions["article_id"].isin(articles["article_id"]).all():
        errors.append("정제된 articles에 없는 article_id가 거래에 있음")
    if sum(len(s) for s in splits.values()) != len(transactions):
        errors.append("분할 행 수 합이 정제된 거래 수와 다름")
    if len(customers) != n_customers_raw:
        errors.append("customers 행 수가 원본과 다름")
    frames = {
        "transactions": transactions,
        "articles": articles,
        "customers": customers,
    }
    for name, cols in KEY_COLUMNS.items():
        nan_cols = [c for c in cols if frames[name][c].isna().any()]
        if nan_cols:
            errors.append(f"{name} 핵심 컬럼 NaN: {nan_cols}")

    empty = [name for name, s in splits.items() if s.empty]
    if empty:
        msg = f"빈 분할: {empty}"
        if dry_run:
            logger.warning("%s (dry run이라 날짜 순서 검증 건너뜀)", msg)
        else:
            errors.append(msg)
    else:
        train, valid, test = splits["train"], splits["valid"], splits["test"]
        if not train["t_dat"].max() < valid["t_dat"].min():
            errors.append("train 최대 날짜 >= valid 최소 날짜")
        if not valid["t_dat"].max() < test["t_dat"].min():
            errors.append("valid 최대 날짜 >= test 최소 날짜")

    if errors:
        raise ValueError("검증 실패: " + "; ".join(errors))
    logger.info("검증 통과")


def save(
    processed_dir: str,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    report: dict,
) -> None:
    """parquet 5개와 preprocess_report.json을 저장한다.

    Args:
        processed_dir: 저장 디렉터리.
        articles: 정제된 articles.
        customers: 정제된 customers.
        splits: {"train", "valid", "test"} -> DataFrame.
        report: 리포트 dict.
    """
    os.makedirs(processed_dir, exist_ok=True)
    articles.to_parquet(os.path.join(processed_dir, "articles.parquet"), index=False)
    customers.to_parquet(os.path.join(processed_dir, "customers.parquet"), index=False)
    for name, df in splits.items():
        path = os.path.join(processed_dir, f"transactions_{name}.parquet")
        # customer_id category 사전(137만 해시)이 row group마다 반복 저장되지 않도록
        # 안 쓰는 category를 지우고 단일 row group으로 쓴다
        # NOTE: 단일 row group이라 부분 읽기 불가. 필요하면 customer_id 정수화
        df = df.assign(customer_id=df["customer_id"].cat.remove_unused_categories())
        df.to_parquet(path, index=False, row_group_size=len(df))
    with open(
        os.path.join(processed_dir, "preprocess_report.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("저장 완료: %s", processed_dir)


def run_preprocess(config: dict) -> dict:
    """전처리 전체를 실행한다. transactions_nrows가 설정되면 dry run(저장 안 함).

    Args:
        config: preprocess config dict.

    Returns:
        리포트 dict.

    Raises:
        ValueError: 정제 또는 검증 실패 시.
    """
    dry_run = config["load"]["transactions_nrows"] is not None
    if dry_run:
        logger.warning(
            "dry run: transactions_nrows=%d, 저장하지 않음",
            config["load"]["transactions_nrows"],
        )
    report = {"dry_run": dry_run}

    articles, customers, transactions = load_raw(config)
    n_customers_raw = len(customers)
    logger.info(
        "raw: 상품 %d, 유저 %d, 거래 %d",
        len(articles),
        n_customers_raw,
        len(transactions),
    )

    articles = clean_articles(articles, config, report)
    customers = clean_customers(customers, config, report)
    transactions = clean_transactions(transactions, articles, customers, report)

    split_cfg = config["split"]
    train, valid, test = split_by_week(
        transactions, split_cfg["test_weeks"], split_cfg["valid_weeks"]
    )
    splits = {"train": train, "valid": valid, "test": test}
    report["splits"] = {}
    for name, df in splits.items():
        info = {
            "rows": len(df),
            "users": int(df["customer_id"].nunique()),
            "articles": int(df["article_id"].nunique()),
            "date_min": str(df["t_dat"].min().date()) if len(df) else None,
            "date_max": str(df["t_dat"].max().date()) if len(df) else None,
        }
        report["splits"][name] = info
        logger.info("split %s: %s", name, info)

    validate(articles, customers, n_customers_raw, transactions, splits, dry_run)
    if dry_run:
        logger.warning("dry run이라 parquet과 리포트를 저장하지 않음")
        return report
    save(config["paths"]["processed_dir"], articles, customers, splits, report)
    return report
