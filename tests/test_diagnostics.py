"""추천 진단 지표와 ALS 실험 조합 테스트.

hit의 재구매/신규 구매 분해, 재구매 추천 비율(상위 k와 중복 추천 처리), 추천 상품
종류 수, 실험 grid 곱집합과 모델 설정별 묶음, ALS 커버/미커버 분할, fallback 포함/제외
hit 분해, 평가 주를 옮긴 재확인 분할을 확인한다.
"""

import pandas as pd
import pytest
from src.evaluation.diagnostics import (
    decompose_hits,
    n_distinct_items,
    purchased_pairs,
    repurchase_rate,
)
from src.pipeline.als import (
    expand_grid,
    group_by_model,
    hit_breakdown,
    shift_eval_split,
    split_by_coverage,
)

TRAIN = pd.DataFrame(
    {
        "customer_id": pd.Categorical(["u1", "u1", "u1", "u2", "u9"]),
        "article_id": pd.array([1, 1, 2, 3, 4], dtype="int32"),
    }
)
GT = {"u1": {1, 5}, "u2": {6}, "u3": {4}}


def test_purchased_pairs_dedup_and_filter():
    pairs = purchased_pairs(TRAIN, ["u1", "u2", "u3"])
    assert sorted(zip(pairs["customer_id"], pairs["article_id"])) == [
        ("u1", 1),
        ("u1", 2),
        ("u2", 3),
    ]


def test_decompose_hits():
    purchased = purchased_pairs(TRAIN, list(GT))
    # u1: 1은 재구매 hit, 5는 신규 hit, 1 중복은 한 번만. u2: hit 없음.
    # u3: 4는 u9가 산 상품이라 u3에겐 신규 hit
    recs = {"u1": [1, 1, 5, 2], "u2": [3], "u3": [4]}
    assert decompose_hits(recs, GT, purchased, k=3) == {
        "hits": 3,
        "repurchase_hits": 1,
        "new_purchase_hits": 2,
    }
    # k=1이면 u1은 1만, u3는 4
    assert decompose_hits(recs, GT, purchased, k=1)["hits"] == 2


def test_repurchase_rate_and_distinct_items():
    purchased = purchased_pairs(TRAIN, list(GT))
    recs = {"u1": [1, 2, 7, 7], "u2": [3, 8]}
    # 상위 3개(u1: 1, 2, 7) + u2(3, 8) = 5개 중 재구매 1, 2, 3
    assert repurchase_rate(recs, purchased, k=3) == 3 / 5
    assert repurchase_rate({}, purchased, k=3) == 0.0
    assert n_distinct_items(recs) == 5


def test_expand_grid_and_group_by_model():
    combos = expand_grid(
        {"filter_already_purchased": [False, True], "train_weeks": [None, 4]}
    )
    assert len(combos) == 4
    assert combos[0] == {"filter_already_purchased": False, "train_weeks": None}
    base = {"factors": 2, "k": 12}
    groups = group_by_model([{**base, **c} for c in combos])
    # filter만 다른 조합은 같은 모델: train_weeks 2종 = 2그룹
    assert [len(g) for g in groups.values()] == [2, 2]
    assert all(len({c["train_weeks"] for c in g}) == 1 for g in groups.values())


def test_split_by_coverage():
    out = split_by_coverage(GT, {"u1": [1], "u9": [2]})  # u9는 정답셋에 없음
    assert out == {"covered": {"u1": {1, 5}}, "uncovered": {"u2": {6}, "u3": {4}}}


def test_hit_breakdown_filled_vs_only():
    purchased = purchased_pairs(TRAIN, list(GT))
    fallback = {"u1": [5], "u2": [6], "u3": [9]}
    als_recs = {"u1": [1, 2]}
    recs = {**fallback, **als_recs}
    ctx = {"ground_truth": GT, "fallback": fallback}
    out = hit_breakdown(ctx, als_recs, recs, purchased, k=12)
    assert out["age_fallback"]["hits"] == 2  # u1의 5, u2의 6
    assert out["als_only"] == {
        "hits": 1,
        "repurchase_hits": 1,
        "new_purchase_hits": 0,
    }
    # 미커버 u2가 받은 fallback hit 1건이 더해진다
    assert out["als_filled"]["hits"] == 2


def test_shift_eval_split():
    train = pd.DataFrame(
        {"customer_id": ["a", "b", "c", "d"], "week_idx": [2, 3, 4, 2]}
    )
    new_train, target = shift_eval_split(train, test_weeks=2, valid_weeks=1)
    assert sorted(target["customer_id"]) == ["a", "d"]
    assert new_train["week_idx"].min() == 3
    with pytest.raises(ValueError):
        shift_eval_split(train, test_weeks=10, valid_weeks=1)
