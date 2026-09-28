"""평가 그룹 라벨링.

역할: 성능을 유저/상품 그룹별로 나눠 보기 위한 라벨을 만든다(cold/warm 유저,
    판매 빈도 구간, 신상품, 메타데이터 Unknown 상품, 나이대).
동작: 거래 기반 라벨은 전부 train만 보고 계산한다(valid/test 정보 누수 방지).
    groupby/value_counts/qcut 같은 pandas 벡터 연산으로 계산한 뒤 dict로 바꾼다.
"""

import pandas as pd

FREQUENCY_LABELS = ["low", "mid", "high"]


def label_user_activity(train_df: pd.DataFrame, cold_threshold: int) -> dict[str, str]:
    """train 구매 건수로 유저를 cold / warm으로 나눈다.

    train에 거래가 없는 유저는 결과에 없다(구매 0회이므로 호출하는 쪽에서 cold로 본다).

    Args:
        train_df: train 거래 (customer_id).
        cold_threshold: 이 값 이하 구매 건수는 cold.

    Returns:
        {customer_id: "cold" | "warm"}.
    """
    counts = train_df["customer_id"].value_counts()
    counts = counts[counts > 0]  # category의 미사용 값 제외
    labels = (counts <= cold_threshold).map({True: "cold", False: "warm"})
    return {str(k): v for k, v in labels.items()}


def label_item_frequency(
    train_df: pd.DataFrame, quantiles: list[float]
) -> dict[int, str]:
    """train 구매자 수 기준 분위로 상품을 low / mid / high로 나눈다.

    Args:
        train_df: train 거래 (customer_id, article_id).
        quantiles: 경계 분위 (예: [0.33, 0.67]).

    Returns:
        {article_id: "low" | "mid" | "high"}. train에 없는 상품은 없다.

    Raises:
        ValueError: 분위 경계가 겹쳐(동률이 너무 많아) 구간을 만들 수 없을 때.
    """
    buyers = (
        train_df[["customer_id", "article_id"]]
        .drop_duplicates()
        .groupby("article_id")
        .size()
    )
    labels = pd.qcut(buyers, q=[0, *quantiles, 1], labels=FREQUENCY_LABELS)
    return labels.astype(str).to_dict()


def label_new_items(train_df: pd.DataFrame, target_df: pd.DataFrame) -> dict[int, bool]:
    """target(valid/test) 상품 중 train에 없던 신상품을 표시한다.

    Args:
        train_df: train 거래 (article_id).
        target_df: valid 또는 test 거래 (article_id).

    Returns:
        {target의 article_id: train에 없었으면 True}.
    """
    target_ids = pd.Series(target_df["article_id"].unique())
    is_new = ~target_ids.isin(train_df["article_id"].unique())
    return dict(zip(target_ids.tolist(), is_new.tolist()))


def label_unknown_metadata(
    articles_df: pd.DataFrame, unknown_label: str
) -> dict[int, bool]:
    """product_group_name이 Unknown인 상품을 표시한다.

    Args:
        articles_df: 정제된 articles (article_id, product_group_name).
        unknown_label: Unknown 라벨.

    Returns:
        {article_id: product_group_name이 Unknown이면 True}.
    """
    is_unknown = articles_df["product_group_name"] == unknown_label
    return dict(zip(articles_df["article_id"].tolist(), is_unknown.tolist()))


def label_age_group(customers_df: pd.DataFrame) -> dict[str, str]:
    """이미 있는 age_group 컬럼을 dict로 바꾼다 (재계산하지 않음).

    Args:
        customers_df: 정제된 customers (customer_id, age_group).

    Returns:
        {customer_id: age_group}.
    """
    return dict(
        zip(
            customers_df["customer_id"].astype(str).tolist(),
            customers_df["age_group"].astype(str).tolist(),
        )
    )
