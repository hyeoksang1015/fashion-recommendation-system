"""ALS 모델 테스트.

유저 3명, 상품 4개의 가짜 train으로 행렬 shape/값(1 + alpha * 횟수)/인덱스 복원,
학습에 없는 유저 제외, 추천 원소 타입(int), 구매 상품 필터 on/off를 확인한다.
"""

import pandas as pd
from src.models.als import build_interaction_matrix, recommend_als, train_als

# u1은 101을 2번 산다. week_idx 2가 train 마지막 주.
TRAIN = pd.DataFrame(
    {
        "customer_id": pd.Categorical(["u1", "u1", "u1", "u2", "u2", "u3", "u3"]),
        "article_id": pd.array([101, 101, 102, 102, 103, 101, 104], dtype="int32"),
        "week_idx": pd.array([2, 3, 2, 2, 5, 2, 3], dtype="int16"),
    }
)
CONFIG = {"factors": 2, "regularization": 0.01, "iterations": 5, "seed": 0}


def _dense(matrix, user_ids, item_ids):
    return pd.DataFrame(matrix.toarray(), index=user_ids, columns=item_ids)


def test_matrix_shape_values_and_ids():
    matrix, user_ids, item_ids = build_interaction_matrix(TRAIN, 0.5, None)
    assert matrix.shape == (3, 4)
    assert matrix.nnz == 6
    dense = _dense(matrix, user_ids, item_ids)
    assert dense.loc["u1", 101] == 1 + 0.5 * 2
    assert dense.loc["u1", 102] == 1 + 0.5 * 1
    assert dense.loc["u2", 101] == 0
    assert list(user_ids) == ["u1", "u2", "u3"]
    assert list(item_ids) == [101, 102, 103, 104]


def test_matrix_train_weeks():
    # 마지막 2주(week 2, 3)만: u2의 103(week 5)이 빠진다
    matrix, _, item_ids = build_interaction_matrix(TRAIN, 1.0, 2)
    assert list(item_ids) == [101, 102, 104]
    assert matrix.nnz == 5


def _recommend(targets, k, filter_purchased):
    matrix, user_ids, item_ids = build_interaction_matrix(TRAIN, 1.0, None)
    model = train_als(matrix, CONFIG)
    return recommend_als(
        model, matrix, user_ids, item_ids, targets, k, filter_purchased
    )


def test_unknown_user_dropped_and_int_items():
    recs = _recommend(["u1", "new_user", "u3"], 3, False)
    assert set(recs) == {"u1", "u3"}
    assert all(type(a) is int for items in recs.values() for a in items)
    assert all(len(items) == 3 for items in recs.values())


def test_filter_already_purchased():
    # u1은 101, 102를 샀다
    kept = _recommend(["u1"], 4, False)["u1"]
    filtered = _recommend(["u1"], 4, True)["u1"]
    assert sorted(kept) == [101, 102, 103, 104]
    assert sorted(filtered) == [103, 104]
