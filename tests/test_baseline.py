"""baseline 파이프라인의 그룹 분할 테스트.

라벨에 없는 유저가 기본 라벨(cold)로 가는지, 상품 필터 후 정답이 빈 유저가 빠지는지,
상품 그룹 평가에서 precision 없이 recall/map만 나오는지 확인한다.
"""

from src.pipeline.baseline import evaluate_baseline, filter_items, split_users

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
