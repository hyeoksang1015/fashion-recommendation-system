"""추천 결과 진단 지표.

역할: 추천이 이미 산 상품(재구매)에 얼마나 기대는지 본다. hit를 재구매 hit와
    신규 구매 hit로 나누고(추천 위치 포함), 추천 전체 중 재구매 상품 비율을
    계산한다.
동작: 추천 dict와 정답 dict를 (customer_id, article_id) 행으로 펼친 뒤 merge로
    hit와 재구매 여부를 찾는다. 같은 상품의 중복 추천은 첫 번째만 쓴다(지표 정의와
    동일). customer_id는 str, article_id는 int64로 맞춘다.
"""

import itertools
import numpy as np
import pandas as pd

KEYS = ["customer_id", "article_id"]


def _to_frame(user_items: dict[str, list[int] | set[int]]) -> pd.DataFrame:
    """{customer_id: 상품 모음}을 순서를 지킨 (customer_id, article_id) 행으로 편다.

    Args:
        user_items: {customer_id: article_id 리스트 또는 집합}.

    Returns:
        customer_id(str), article_id(int64) 컬럼 DataFrame.
    """
    lengths = np.fromiter(map(len, user_items.values()), dtype=np.int64)
    return pd.DataFrame(
        {
            "customer_id": np.repeat(
                np.asarray(list(user_items), dtype=object), lengths
            ),
            "article_id": np.fromiter(
                itertools.chain.from_iterable(user_items.values()),
                dtype=np.int64,
                count=int(lengths.sum()),
            ),
        }
    )


def purchased_pairs(train_df: pd.DataFrame, users: list[str]) -> pd.DataFrame:
    """users가 train에서 산 (customer_id, article_id) 쌍을 중복 없이 만든다.

    Args:
        train_df: train 거래 (customer_id, article_id).
        users: 대상 customer_id 목록.

    Returns:
        customer_id(str), article_id(int64) 컬럼 DataFrame.
    """
    pairs = train_df.loc[train_df["customer_id"].isin(users), KEYS].drop_duplicates()
    return pd.DataFrame(
        {
            "customer_id": pairs["customer_id"].astype(str).to_numpy(dtype=object),
            "article_id": pairs["article_id"].to_numpy(dtype=np.int64),
        }
    )


def _is_purchased(frame: pd.DataFrame, purchased: pd.DataFrame) -> np.ndarray:
    """frame 각 행이 purchased 쌍에 있는지 표시한다.

    Args:
        frame: customer_id, article_id 컬럼 DataFrame.
        purchased: purchased_pairs 결과.

    Returns:
        frame 행 순서의 bool 배열.
    """
    index = pd.MultiIndex.from_frame(purchased[KEYS])
    return pd.MultiIndex.from_frame(frame[KEYS]).isin(index)


def _top_k(recs: dict[str, list[int]], k: int) -> pd.DataFrame:
    """추천을 상위 k개로 자르고 유저 안 중복 상품은 첫 번째만 남긴다.

    Args:
        recs: {customer_id: 순위순 추천 리스트}.
        k: 자를 순위.

    Returns:
        customer_id, article_id, rank(1부터 시작하는 추천 위치) 컬럼 DataFrame.
    """
    frame = _to_frame(recs)
    frame["rank"] = frame.groupby("customer_id", sort=False).cumcount() + 1
    return frame[frame["rank"] <= k].drop_duplicates(KEYS)


def hit_ranks(
    recs: dict[str, list[int]],
    ground_truth: dict[str, set[int]],
    purchased: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """상위 k개 추천 중 hit 행을 추천 위치, 재구매 여부와 함께 반환한다.

    Args:
        recs: {customer_id: 순위순 추천 리스트}.
        ground_truth: {customer_id: 정답 집합}.
        purchased: purchased_pairs 결과 (train에서 이미 산 쌍).
        k: 자를 순위.

    Returns:
        customer_id, article_id, rank(1부터), repurchase(bool) 컬럼 DataFrame.
        같은 상품의 중복 추천은 첫 위치만 남는다.
    """
    hits = _top_k(recs, k).merge(_to_frame(ground_truth), on=KEYS)
    hits["repurchase"] = _is_purchased(hits, purchased)
    return hits


def decompose_hits(
    recs: dict[str, list[int]],
    ground_truth: dict[str, set[int]],
    purchased: pd.DataFrame,
    k: int,
) -> dict[str, int]:
    """상위 k개 hit를 재구매 hit와 신규 구매 hit로 나눈다.

    Args:
        recs: {customer_id: 순위순 추천 리스트}.
        ground_truth: {customer_id: 정답 집합}.
        purchased: purchased_pairs 결과 (train에서 이미 산 쌍).
        k: 자를 순위.

    Returns:
        {"hits": 전체 hit 수, "repurchase_hits": train에서 산 상품 hit 수,
        "new_purchase_hits": train에서 사지 않은 상품 hit 수}.
    """
    hits = hit_ranks(recs, ground_truth, purchased, k)
    repurchase = int(hits["repurchase"].sum())
    return {
        "hits": len(hits),
        "repurchase_hits": repurchase,
        "new_purchase_hits": len(hits) - repurchase,
    }


def repurchase_rate(
    recs: dict[str, list[int]], purchased: pd.DataFrame, k: int
) -> float:
    """상위 k개 추천 중 train에서 이미 산 상품의 비율.

    Args:
        recs: {customer_id: 순위순 추천 리스트}.
        purchased: purchased_pairs 결과.
        k: 자를 순위.

    Returns:
        0~1 비율. 추천이 없으면 0.
    """
    frame = _top_k(recs, k)
    if frame.empty:
        return 0.0
    return float(_is_purchased(frame, purchased).mean())


def n_distinct_items(recs: dict[str, list[int]]) -> int:
    """추천에 나온 상품 종류 수.

    Args:
        recs: {customer_id: 추천 리스트}.

    Returns:
        distinct article_id 수.
    """
    return int(_to_frame(recs)["article_id"].nunique())
