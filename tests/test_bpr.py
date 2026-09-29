"""BPR 모델 테스트.

test_als와 같은 가짜 train(유저 3, 상품 4)으로 이진 행렬(재구매도 1), 학습된 factor
shape, 추천의 학습 유저 제한과 article_id(int) 복원, 구매 상품 필터를 확인한다.
"""

import numpy as np
from src.models.als import recommend_als
from src.models.bpr import build_binary_matrix, train_bpr
from tests.test_als import TRAIN

CONFIG = {
    "factors": 2,
    "learning_rate": 0.05,
    "regularization": 0.01,
    "iterations": 20,
    "seed": 0,
    "num_threads": 1,
}


def test_binary_matrix():
    matrix, user_ids, item_ids = build_binary_matrix(TRAIN, None)
    assert matrix.shape == (3, 4)
    assert matrix.nnz == 6
    # u1은 101을 2번 샀지만 값은 1
    assert np.all(matrix.data == 1.0)
    assert list(user_ids) == ["u1", "u2", "u3"]
    assert list(item_ids) == [101, 102, 103, 104]


def test_train_shapes_and_deterministic():
    matrix, _, _ = build_binary_matrix(TRAIN, None)
    model = train_bpr(matrix, CONFIG)
    # implicit BPR은 factor 뒤에 상품 bias 1차원을 붙인다
    assert model.user_factors.shape == (3, CONFIG["factors"] + 1)
    assert model.item_factors.shape == (4, CONFIG["factors"] + 1)
    again = train_bpr(matrix, CONFIG)
    assert np.allclose(model.item_factors, again.item_factors)


def _recommend(targets, k, filter_purchased):
    matrix, user_ids, item_ids = build_binary_matrix(TRAIN, None)
    model = train_bpr(matrix, CONFIG)
    return recommend_als(
        model, matrix, user_ids, item_ids, targets, k, filter_purchased
    )


def test_recommend_restores_int_ids():
    recs = _recommend(["u1", "new_user", "u3"], 3, False)
    assert set(recs) == {"u1", "u3"}
    assert all(type(a) is int for items in recs.values() for a in items)
    assert all(set(items) <= {101, 102, 103, 104} for items in recs.values())


def test_filter_already_purchased():
    # u1은 101, 102를 샀다
    assert sorted(_recommend(["u1"], 4, False)["u1"]) == [101, 102, 103, 104]
    assert sorted(_recommend(["u1"], 4, True)["u1"]) == [103, 104]
