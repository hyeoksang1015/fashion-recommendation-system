"""상품 인기도 집계.

역할: 인기 baseline과 fallback 추천에 쓸 상품 순위를 train 거래로만 계산한다.
동작: 인기도는 거래 건수가 아니라 distinct 구매자 수로 센다(재구매가 많은 유저 한 명이
    인기도를 부풀리지 않도록). (customer_id, article_id) 중복을 지운 뒤 groupby로 센다.
    순위는 구매자 수 내림차순, 동률이면 article_id 오름차순으로 고정한다.
    상위 k개로 자르지 않고 전체 순위를 반환한다(자르기는 fallback.py에서).
"""

import pandas as pd

COUNT_COL = "n_buyers"


def _last_weeks(train_df: pd.DataFrame, n_weeks: int) -> pd.DataFrame:
    """train의 마지막 n_weeks주 거래만 남긴다.

    week_idx는 작을수록 최근이다. train은 valid/test 뒤에 오므로 train의 최소
    week_idx부터 n_weeks주를 쓴다.

    Args:
        train_df: week_idx가 있는 train 거래.
        n_weeks: 사용할 주 수.

    Returns:
        마지막 n_weeks주 거래.

    Raises:
        ValueError: n_weeks가 1 미만일 때.
    """
    if n_weeks < 1:
        raise ValueError(f"n_weeks는 1 이상이어야 함: {n_weeks}")
    last_week = train_df["week_idx"].min()
    return train_df[train_df["week_idx"] < last_week + n_weeks]


def _count_buyers(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """keys별 distinct 구매자 수를 세서 정렬한다.

    Args:
        df: customer_id와 keys 컬럼이 있는 거래.
        keys: 집계 기준 컬럼 (마지막이 article_id).

    Returns:
        keys + n_buyers 컬럼 DataFrame. keys[:-1] 오름차순, n_buyers 내림차순,
        article_id 오름차순.
    """
    counts = (
        df[["customer_id", *keys]]
        .drop_duplicates()
        .groupby(keys, observed=True)
        .size()
        .rename(COUNT_COL)
        .reset_index()
    )
    by = [*keys[:-1], COUNT_COL, "article_id"]
    ascending = [True] * (len(keys) - 1) + [False, True]
    return counts.sort_values(by, ascending=ascending, ignore_index=True)


def compute_overall_popularity(train_df: pd.DataFrame) -> pd.DataFrame:
    """전체 train 기간의 article_id별 구매자 수 순위.

    Args:
        train_df: train 거래 (customer_id, article_id).

    Returns:
        article_id, n_buyers 컬럼. 구매자 수 내림차순 전체 순위.
    """
    return _count_buyers(train_df, ["article_id"])


def compute_recent_popularity(train_df: pd.DataFrame, n_weeks: int) -> pd.DataFrame:
    """train 마지막 n_weeks주의 article_id별 구매자 수 순위.

    Args:
        train_df: train 거래 (customer_id, article_id, week_idx).
        n_weeks: 사용할 최근 주 수 (config recent_week_window).

    Returns:
        article_id, n_buyers 컬럼. 구매자 수 내림차순 전체 순위.

    Raises:
        ValueError: n_weeks가 1 미만일 때.
    """
    return _count_buyers(_last_weeks(train_df, n_weeks), ["article_id"])


def compute_age_group_popularity(
    train_df: pd.DataFrame, customers_df: pd.DataFrame, n_weeks: int
) -> pd.DataFrame:
    """train 마지막 n_weeks주의 age_group별 article_id 구매자 수 순위.

    최근 인기 baseline(1주)보다 긴 기간(기본 4주)을 쓴다. 나이대로 나누면 그룹마다
    구매자가 적어져 1주로는 순위가 불안정하기 때문이다. 4주는 EDA 5.4절에서
    검증한 값이다.

    Args:
        train_df: train 거래 (customer_id, article_id, week_idx).
        customers_df: customer_id, age_group 컬럼이 있는 정제된 customers.
        n_weeks: 사용할 최근 주 수 (config age_group_week_window).

    Returns:
        age_group, article_id, n_buyers 컬럼. age_group 안에서 구매자 수 내림차순.
        age_group은 customers의 category를 유지하므로 구매가 없는 그룹도
        category에는 남는다.

    Raises:
        ValueError: n_weeks가 1 미만일 때.
    """
    recent = _last_weeks(train_df, n_weeks)
    age_group = customers_df.set_index("customer_id")["age_group"]
    recent = recent.assign(
        age_group=pd.Categorical(
            recent["customer_id"].astype(str).map(age_group),
            categories=_categories(customers_df["age_group"]),
        )
    )
    return _count_buyers(recent, ["age_group", "article_id"])


def _categories(age_group: pd.Series) -> list:
    """age_group의 전체 라벨 목록 (category면 사용하지 않는 라벨 포함).

    Args:
        age_group: customers의 age_group 컬럼.

    Returns:
        라벨 목록.
    """
    if isinstance(age_group.dtype, pd.CategoricalDtype):
        return list(age_group.cat.categories)
    return sorted(age_group.dropna().unique())
