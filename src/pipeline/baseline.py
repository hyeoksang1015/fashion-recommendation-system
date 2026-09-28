"""인기 기반 baseline 3종 평가 파이프라인.

역할: overall(전체 기간 인기), recent(최근 1주 인기), age_fallback(나이대별 최근 인기)
    추천을 만들어 평가 분할(기본 valid)에서 전체 지표와 그룹별 지표를 낸다.
동작:
    1. 전처리 parquet(train, 평가 분할, customers, articles)을 읽는다.
    2. train만으로 인기도와 fallback 테이블, 그룹 라벨을 계산한다(누수 방지).
    3. 평가 분할의 구매 유저에게 baseline별 top-k 추천을 준다.
    4. 유저 그룹(활동, 나이대)은 정답셋 유저를 걸러서, 상품 그룹(판매 빈도, 신상품,
       메타데이터 Unknown)은 정답 상품을 걸러서 같은 evaluate_users로 평가한다.
    5. 결과를 {output_dir}/baseline_results.json에 저장한다.
"""

import json
import logging
import os
import pandas as pd
from src.evaluation.ground_truth import build_ground_truth
from src.evaluation.metrics import evaluate_users
from src.features.fallback import build_fallback_table
from src.features.groups import (
    label_age_group,
    label_item_frequency,
    label_new_items,
    label_unknown_metadata,
    label_user_activity,
)
from src.features.popularity import (
    compute_age_group_popularity,
    compute_overall_popularity,
    compute_recent_popularity,
)
from src.utils.config import load_config

logger = logging.getLogger(__name__)

COLD = "cold"


def build_recommendations(
    users: list[str],
    train: pd.DataFrame,
    customers: pd.DataFrame,
    features_cfg: dict,
) -> dict[str, dict[str, list[int]]]:
    """baseline 3종의 유저별 top-k 추천을 만든다.

    Args:
        users: 추천을 줄 customer_id 목록 (평가 분할의 구매 유저).
        train: train 거래.
        customers: 정제된 customers (customer_id, age_group).
        features_cfg: features config.

    Returns:
        {baseline 이름: {customer_id: [article_id, ...]}}.
    """
    k = features_cfg["top_k"]
    unknown = features_cfg["unknown_label"]
    overall_pop = compute_overall_popularity(train)
    recent_pop = compute_recent_popularity(train, features_cfg["recent_week_window"])
    age_pop = compute_age_group_popularity(
        train, customers, features_cfg["age_group_week_window"]
    )
    fallback = build_fallback_table(overall_pop, age_pop, k, unknown)
    age_of = label_age_group(customers)

    overall_top = overall_pop["article_id"].head(k).tolist()
    recent_top = recent_pop["article_id"].head(k).tolist()
    return {
        "overall": dict.fromkeys(users, overall_top),
        "recent": dict.fromkeys(users, recent_top),
        # customers에 없는 유저와 나이 결측 유저는 unknown_label 목록(전체 인기)
        "age_fallback": {u: fallback[age_of.get(u, unknown)] for u in users},
    }


def split_users(
    ground_truth: dict[str, set[int]], labels: dict[str, str], default: str
) -> dict[str, dict[str, set[int]]]:
    """정답셋 유저를 라벨별로 나눈다.

    Args:
        ground_truth: {customer_id: 정답 집합}.
        labels: {customer_id: 그룹 라벨}.
        default: labels에 없는 유저의 라벨.

    Returns:
        {라벨: 그 라벨 유저만 남은 ground_truth}.
    """
    groups: dict[str, dict[str, set[int]]] = {}
    for user, relevant in ground_truth.items():
        groups.setdefault(labels.get(user, default), {})[user] = relevant
    return groups


def filter_items(
    ground_truth: dict[str, set[int]], keep: set[int]
) -> dict[str, set[int]]:
    """정답 집합을 keep 상품으로 거르고, 남은 정답이 없는 유저는 뺀다.

    Args:
        ground_truth: {customer_id: 정답 집합}.
        keep: 남길 article_id 집합.

    Returns:
        걸러진 ground_truth.
    """
    filtered = {user: relevant & keep for user, relevant in ground_truth.items()}
    return {user: relevant for user, relevant in filtered.items() if relevant}


def _score(
    recs: dict[str, list[int]],
    ground_truth: dict[str, set[int]],
    k: int,
    metrics: list[str],
) -> dict:
    """evaluate_users 결과에 평가 유저 수를 붙인다.

    Args:
        recs: {customer_id: 추천 리스트}.
        ground_truth: 평가할 정답셋.
        k: 자를 순위.
        metrics: 지표 이름 목록.

    Returns:
        {지표: 점수, "n_users": 유저 수}. 정답셋이 비면 {"n_users": 0}.
    """
    if not ground_truth:
        return {"n_users": 0}
    scores = evaluate_users(recs, ground_truth, k, metrics)
    return {**scores, "n_users": len(ground_truth)}


def evaluate_baseline(
    recs: dict[str, list[int]],
    ground_truth: dict[str, set[int]],
    user_groups: dict[str, dict[str, dict[str, set[int]]]],
    item_groups: dict[str, dict[str, set[int]]],
    k: int,
    metrics: list[str],
    item_metrics: list[str],
) -> dict:
    """baseline 하나를 전체 + 유저 그룹 + 상품 그룹으로 평가한다.

    Args:
        recs: {customer_id: 추천 리스트}.
        ground_truth: 전체 정답셋.
        user_groups: {그룹 종류: {라벨: 유저가 걸러진 ground_truth}}.
        item_groups: {그룹 이름: 정답 상품이 걸러진 ground_truth}.
        k: 자를 순위.
        metrics: 전체/유저 그룹 지표.
        item_metrics: 상품 그룹 지표.

    Returns:
        {"all": {...}, "by_<유저 그룹>": {라벨: {...}}, <상품 그룹>: {...}}.
    """
    result = {"all": _score(recs, ground_truth, k, metrics)}
    for kind, groups in user_groups.items():
        result[kind] = {
            label: _score(recs, gt, k, metrics) for label, gt in sorted(groups.items())
        }
    for name, gt in item_groups.items():
        result[name] = _score(recs, gt, k, item_metrics)
    return result


def run_baseline(config: dict) -> dict:
    """baseline 3종을 평가하고 결과를 json으로 저장한다.

    Args:
        config: baseline config (configs/baseline.yaml).

    Returns:
        {baseline 이름: 평가 결과}.
    """
    features_cfg = load_config(config["features_config"])
    eval_cfg = load_config(config["evaluation_config"])
    k, metrics = eval_cfg["k"], eval_cfg["metrics"]
    split = config["target_split"]
    processed = config["processed_dir"]

    train = pd.read_parquet(
        os.path.join(processed, "transactions_train.parquet"),
        columns=["customer_id", "article_id", "week_idx"],
    )
    target = pd.read_parquet(
        os.path.join(processed, f"transactions_{split}.parquet"),
        columns=["customer_id", "article_id"],
    )
    customers = pd.read_parquet(
        os.path.join(processed, "customers.parquet"),
        columns=["customer_id", "age_group"],
    )
    articles = pd.read_parquet(
        os.path.join(processed, "articles.parquet"),
        columns=["article_id", "product_group_name"],
    )

    ground_truth = build_ground_truth(target)
    logger.info("%s 정답셋: 유저 %d명", split, len(ground_truth))
    all_recs = build_recommendations(list(ground_truth), train, customers, features_cfg)

    # train에 한 번도 없던 유저는 label_user_activity 결과에 없다.
    # 구매 이력이 0회이므로 cold(구매 <= threshold)로 본다.
    activity = label_user_activity(train, features_cfg["cold_warm_threshold"])
    user_groups = {
        "by_user_activity": split_users(ground_truth, activity, COLD),
        "by_age_group": split_users(
            ground_truth, label_age_group(customers), features_cfg["unknown_label"]
        ),
    }

    frequency = label_item_frequency(train, features_cfg["item_frequency_quantiles"])
    new_items = label_new_items(train, target)
    unknown_meta = label_unknown_metadata(articles, features_cfg["unknown_label"])
    # 판매 빈도 라벨은 train 상품에만 있으므로 신상품은 low/mid/high 어디에도 없다
    item_groups = {
        f"item_frequency_{label}": filter_items(
            ground_truth, {a for a, lab in frequency.items() if lab == label}
        )
        for label in ("low", "mid", "high")
    }
    item_groups["new_items"] = filter_items(
        ground_truth, {a for a, is_new in new_items.items() if is_new}
    )
    item_groups["unknown_metadata"] = filter_items(
        ground_truth, {a for a, is_unk in unknown_meta.items() if is_unk}
    )

    results = {}
    for name, recs in all_recs.items():
        results[name] = evaluate_baseline(
            recs,
            ground_truth,
            user_groups,
            item_groups,
            k,
            metrics,
            config["item_group_metrics"],
        )
        logger.info("%s: %s", name, results[name]["all"])

    os.makedirs(config["output_dir"], exist_ok=True)
    path = os.path.join(config["output_dir"], "baseline_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"split": split, "k": k, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("저장 완료: %s", path)
    return results
