"""BPR(implicit) 협업 필터링 모델.

역할: train 거래로 user x item 이진 행렬(구매 여부)을 만들고 implicit의
    BayesianPersonalizedRanking을 학습한다.
동작:
    1. 행렬은 ALS의 build_interaction_matrix를 alpha=0으로 불러 만든다. 값이
       1 + 0 * 횟수 = 1이 되어 구매 여부만 남는다. implicit BPR은 비제로 위치만
       양성 표본으로 쓰고 값은 보지 않으므로(값을 바꿔도 학습 결과가 같음을 확인),
       구매 횟수는 BPR에 반영되지 않는다. ALS와의 차이점이다.
    2. implicit 0.7의 fit은 ALS와 같이 user x item CSR을 받는다.
    3. 추천은 ALS와 같은 recommend_als(implicit 행렬 분해 모델 공통)를 쓴다.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.bpr import BayesianPersonalizedRanking
from src.models.als import build_interaction_matrix

BINARY_ALPHA = 0.0  # confidence = 1 + 0 * 횟수 = 1 (구매 여부)


def build_binary_matrix(
    train_df: pd.DataFrame, train_weeks: int | None
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """user x item 구매 여부(0/1) 행렬을 만든다.

    Args:
        train_df: train 거래 (customer_id, article_id, week_idx).
        train_weeks: None이면 train 전체, 정수면 train 마지막 N주만 쓴다.

    Returns:
        (matrix, user_ids, item_ids). matrix 값은 모두 1.0 (float32 CSR).

    Raises:
        ValueError: train_weeks가 1 미만일 때.
    """
    return build_interaction_matrix(train_df, BINARY_ALPHA, train_weeks)


def train_bpr(matrix: sp.csr_matrix, config: dict) -> BayesianPersonalizedRanking:
    """BPR을 학습한다.

    Args:
        matrix: build_binary_matrix의 user x item 행렬.
        config: bpr config (factors, learning_rate, regularization, iterations,
            seed, num_threads).

    Returns:
        학습된 BayesianPersonalizedRanking.
    """
    model = BayesianPersonalizedRanking(
        factors=config["factors"],
        learning_rate=config["learning_rate"],
        regularization=config["regularization"],
        iterations=config["iterations"],
        random_state=config["seed"],
        num_threads=config["num_threads"],
        use_gpu=False,
    )
    model.fit(matrix, show_progress=False)
    return model
