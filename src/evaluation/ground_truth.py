"""평가용 정답셋 생성.

역할: valid/test 거래에서 유저별 실제 구매 상품 집합을 만든다.
동작: customer_id로 groupby해 article_id를 set으로 모은다. 같은 상품을 여러 번 사도
    한 번으로 본다(binary relevance). article_id는 parquet의 정수 dtype 그대로 쓰며,
    set 원소는 파이썬 int가 된다(numpy int32와 set 조회 결과가 같다).
"""

import pandas as pd


def build_ground_truth(transactions_df: pd.DataFrame) -> dict[str, set[int]]:
    """customer_id별 실제 구매 article_id 집합을 만든다.

    Args:
        transactions_df: valid 또는 test 거래 DataFrame
            (customer_id, article_id 컬럼 필요).

    Returns:
        {customer_id: {article_id, ...}}. article_id는 int.
    """
    grouped = (
        transactions_df["article_id"]
        .groupby(transactions_df["customer_id"], observed=True, sort=False)
        .agg(set)
    )
    return {str(k): v for k, v in grouped.items()}
