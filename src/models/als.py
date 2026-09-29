"""ALS(implicit) 협업 필터링 모델.

역할: train 거래로 user x item 신뢰도 행렬을 만들고, implicit의 ALS를 학습해
    유저별 top-k 추천을 만든다.
동작:
    1. (customer_id, article_id)별 구매 횟수를 groupby size로 세고, 값을
       confidence = 1 + alpha * 횟수로 둔다. 인덱스는 pandas Categorical codes다.
    2. implicit 0.7의 fit은 user x item CSR을 받는다. implicit은 행렬 값을 그대로
       confidence c_ui로 쓰므로(선호 p_ui = 1), 모델의 alpha는 기본값 1.0으로 둔다.
       alpha는 행렬을 만들 때 한 번만 적용된다.
    3. 추천은 학습 유저 전체를 한 번의 배치 recommend 호출로 만든다. 학습에 없는
       유저는 결과에서 빠지고, fallback은 호출하는 쪽에서 채운다.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase
from threadpoolctl import threadpool_limits


def build_interaction_matrix(
    train_df: pd.DataFrame, alpha: float, train_weeks: int | None
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """user x item confidence 행렬을 만든다.

    Args:
        train_df: train 거래 (customer_id, article_id, week_idx).
        alpha: confidence = 1 + alpha * 구매 횟수의 alpha.
        train_weeks: None이면 train 전체, 정수면 train 마지막 N주만 쓴다
            (week_idx가 작을수록 최근이므로 최소 week_idx부터 N주).

    Returns:
        (matrix, user_ids, item_ids). matrix는 float32 CSR (유저 수, 상품 수)이고
        matrix[i, j]는 user_ids[i]가 item_ids[j]를 산 confidence다.
        user_ids는 customer_id(str), item_ids는 article_id(int) 배열이다.

    Raises:
        ValueError: train_weeks가 1 미만일 때.
    """
    df = train_df
    if train_weeks is not None:
        if train_weeks < 1:
            raise ValueError(f"train_weeks는 1 이상이어야 함: {train_weeks}")
        df = df[df["week_idx"] < df["week_idx"].min() + train_weeks]

    counts = df.groupby(["customer_id", "article_id"], observed=True).size()
    users = pd.Categorical(
        counts.index.get_level_values("customer_id")
    ).remove_unused_categories()
    items = pd.Categorical(
        counts.index.get_level_values("article_id")
    ).remove_unused_categories()
    values = (1 + alpha * counts.to_numpy()).astype(np.float32)
    matrix = sp.csr_matrix(
        (values, (users.codes, items.codes)),
        shape=(len(users.categories), len(items.categories)),
    )
    user_ids = users.categories.astype(str).to_numpy()
    item_ids = items.categories.to_numpy()
    return matrix, user_ids, item_ids


def train_als(matrix: sp.csr_matrix, config: dict) -> AlternatingLeastSquares:
    """ALS를 학습한다.

    Args:
        matrix: build_interaction_matrix의 user x item confidence 행렬.
        config: als config (factors, regularization, iterations, seed).

    Returns:
        학습된 AlternatingLeastSquares.
    """
    model = AlternatingLeastSquares(
        factors=config["factors"],
        regularization=config["regularization"],
        iterations=config["iterations"],
        random_state=config["seed"],
        use_gpu=False,
    )
    # implicit 권고: BLAS 내부 스레드와 implicit 스레드가 겹치면 크게 느려진다
    with threadpool_limits(1, "blas"):
        model.fit(matrix, show_progress=False)
    return model


def recommend_als(
    model: MatrixFactorizationBase,
    matrix: sp.csr_matrix,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    target_customers: list[str],
    k: int,
    filter_already_purchased: bool,
) -> dict[str, list[int]]:
    """학습 유저에게 top-k 추천을 만든다.

    implicit CPU 행렬 분해 모델(ALS, BPR)의 공통 recommend를 쓴다.

    Args:
        model: 학습된 ALS 또는 BPR.
        matrix: 학습에 쓴 user x item 행렬 (구매 상품 필터용).
        user_ids: 행렬 행 순서의 customer_id 배열.
        item_ids: 행렬 열 순서의 article_id 배열.
        target_customers: 추천을 줄 customer_id 목록.
        k: 추천 개수.
        filter_already_purchased: True면 train에서 산 상품을 추천에서 뺀다.

    Returns:
        {customer_id: [article_id(int), ...]}. 학습 유저에 없는 target은 빠진다.
    """
    targets = np.asarray(target_customers, dtype=object)
    rows = pd.Index(user_ids).get_indexer(targets)
    known = rows >= 0
    rows = rows[known]
    if len(rows) == 0:
        return {}
    with threadpool_limits(1, "blas"):
        ids, scores = model.recommend(
            rows,
            matrix[rows],
            N=k,
            filter_already_liked_items=filter_already_purchased,
        )
    # NOTE: 필터 후 남은 상품이 k개보다 적으면 implicit이 걸러진 상품을 최저 점수
    # (float32 최솟값)로 뒤에 붙인다. 그 자리는 버린다.
    valid = scores > np.finfo(scores.dtype).min
    recs = item_ids[ids]
    return {
        user: row[ok].tolist() for user, row, ok in zip(targets[known], recs, valid)
    }
