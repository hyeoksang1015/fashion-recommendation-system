"""transactions(거래) 정제 함수 모음.

역할: 정제된 상품에 해당하는 거래만 남기고, 시간 분할에 쓸 week_idx를 붙인다.
동작: article_id를 isin으로 필터링하고, 원본 최대 날짜부터 거꾸로 7일씩 잘라
    week_idx를 계산한다(0 = 마지막 7일). 중복 구매, 두 판매 채널, 가격은 그대로
    둔다.
"""

import pandas as pd


def filter_valid_articles(
    df: pd.DataFrame, valid_article_ids: pd.Series
) -> pd.DataFrame:
    """정제된 articles에 없는 상품의 거래를 제거한다.

    Args:
        df: transactions DataFrame.
        valid_article_ids: 남길 article_id.

    Returns:
        유효 상품 거래만 남은 DataFrame.
    """
    return df[df["article_id"].isin(valid_article_ids)]


def count_unknown_customers(df: pd.DataFrame, customer_ids: pd.Series) -> int:
    """customers에 없는 customer_id의 고유 개수를 센다 (제거하지 않음).

    Args:
        df: transactions DataFrame.
        customer_ids: customers의 customer_id.

    Returns:
        customers에 없는 고유 customer_id 수.
    """
    tx_ids = pd.Series(df["customer_id"].unique())
    return int((~tx_ids.isin(customer_ids)).sum())


def add_week_idx(df: pd.DataFrame, max_date: pd.Timestamp) -> pd.DataFrame:
    """종료일부터 거꾸로 7일씩 자른 week_idx(int16)를 추가한다. 0 = 마지막 7일.

    Args:
        df: transactions DataFrame.
        max_date: 원본 거래의 최대 날짜.

    Returns:
        week_idx 컬럼이 추가된 DataFrame.
    """
    out = df.copy()
    out["week_idx"] = ((max_date - out["t_dat"]).dt.days // 7).astype("int16")
    return out
