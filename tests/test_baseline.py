"""baseline 파이프라인의 그룹 분할 테스트.

라벨에 없는 유저가 기본 라벨(cold)로 가는지, 상품 필터 후 정답이 빈 유저가 빠지는지,
상품 그룹 평가에서 precision 없이 recall/map만 나오는지 확인한다. age_fallback
집계 기간을 바꾸면 나이대 목록이 바뀌고, 나이 결측/미등록 유저는 전체 인기를 받는지
확인한다.
"""

import pandas as pd
from src.features.popularity import compute_overall_popularity
from src.pipeline.baseline import (
    build_age_fallback,
    evaluate_baseline,
    filter_items,
    split_users,
)

GT = {"u1": {1, 2}, "u2": {3}, "u3": {4}}


def test_split_users_default_cold():
    labels = {"u1": "warm", "u2": "cold"}  # u3는 train에 없음
    out = split_users(GT, labels, "cold")
    assert out == {"warm": {"u1": {1, 2}}, "cold": {"u2": {3}, "u3": {4}}}


def test_filter_items_drops_empty_users():
    assert filter_items(GT, {2, 4}) == {"u1": {2}, "u3": {4}}


def test_evaluate_baseline_structure():
    recs = dict.fromkeys(GT, [2, 3, 9])
    out = evaluate_baseline(
        recs,
        GT,
        {"by_user_activity": split_users(GT, {"u1": "warm"}, "cold")},
        {"new_items": filter_items(GT, {4}), "empty": {}},
        k=3,
        metrics=["precision", "recall"],
        item_metrics=["recall", "map"],
    )
    assert out["all"]["n_users"] == 3
    assert out["by_user_activity"]["warm"]["recall"] == 0.5  # u1: {1,2} 중 2
    assert out["by_user_activity"]["cold"]["recall"] == 0.5  # u2 1, u3 0
    assert set(out["new_items"]) == {"recall", "map", "n_users"}
    assert out["new_items"]["recall"] == 0
    assert out["empty"] == {"n_users": 0}


def test_build_age_fallback_window():
    # week 2(최근): u1이 10을 산다. week 3: u2, u3가 11을 산다 (모두 20s)
    train = pd.DataFrame(
        {
            "customer_id": pd.Categorical(["u1", "u2", "u3"]),
            "article_id": pd.array([10, 11, 11], dtype="int32"),
            "week_idx": pd.array([2, 3, 3], dtype="int16"),
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3", "u4"],
            "age_group": pd.Categorical(
                ["20s", "20s", "20s", "Unknown"], categories=["20s", "Unknown"]
            ),
        }
    )
    cfg = {"top_k": 2, "unknown_label": "Unknown"}
    overall = compute_overall_popularity(train)
    users = ["u1", "u4", "u9"]  # u9는 customers에 없음

    one = build_age_fallback(users, train, customers, overall, cfg, 1)
    two = build_age_fallback(users, train, customers, overall, cfg, 2)
    assert one["u1"] == [10, 11]  # 1주: 10만 팔림, 전체 인기로 채움
    assert two["u1"] == [11, 10]  # 2주: 11 구매자 2명
    assert one["u4"] == one["u9"] == [11, 10]  # 전체 인기
