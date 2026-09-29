"""견고성 확인 함수 테스트.

paired bootstrap(같은 입력이면 차이 0, 신뢰구간이 평균 차이 포함, seed 재현),
hit 순위 계산(중복 추천은 첫 위치, 재구매 표시), hit 요약, 유저별 지표와 평균의
일치, 시드 요약을 확인한다.
"""

import numpy as np
import pandas as pd
import pytest
from src.evaluation.diagnostics import hit_ranks, purchased_pairs
from src.evaluation.metrics import evaluate_users, evaluate_users_per_user
from src.evaluation.significance import paired_bootstrap
from src.pipeline.robustness import hit_profile, summarize_seeds

TRAIN = pd.DataFrame(
    {
        "customer_id": pd.Categorical(["u1", "u1", "u2"]),
        "article_id": pd.array([1, 2, 3], dtype="int32"),
    }
)
GT = {"u1": {1, 5}, "u2": {6, 3}, "u3": {4}}


def test_bootstrap_same_input_zero():
    scores = np.array([0.0, 0.5, 1.0, 0.25])
    out = paired_bootstrap(scores, scores, n_resamples=200, seed=0, ci_level=0.95)
    assert out["mean_diff"] == 0
    assert out["ci_low"] == 0 and out["ci_high"] == 0


def test_bootstrap_ci_contains_mean_and_reproducible():
    rng = np.random.default_rng(1)
    a = rng.random(500)
    b = a + rng.normal(0.05, 0.1, 500)
    out = paired_bootstrap(a, b, n_resamples=500, seed=3, ci_level=0.95)
    assert out["ci_low"] <= out["mean_diff"] <= out["ci_high"]
    assert out["mean_diff"] == pytest.approx((b - a).mean())
    assert out == paired_bootstrap(a, b, n_resamples=500, seed=3, ci_level=0.95)


def test_bootstrap_invalid_input():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(3), np.zeros(4), 10, 0, 0.95)
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(3), np.zeros(3), 10, 0, 1.5)


def test_hit_ranks():
    purchased = purchased_pairs(TRAIN, list(GT))
    # u1: 1은 순위 2의 재구매 hit(순위 1은 miss), 1 중복은 무시, 5는 순위 4 신규 hit
    # u2: 3은 순위 1 재구매 hit, 6은 순위 3 신규 hit(k=2면 밖). u3: hit 없음
    recs = {"u1": [9, 1, 1, 5], "u2": [3, 7, 6], "u3": [8]}
    hits = hit_ranks(recs, GT, purchased, k=4)
    got = sorted(
        zip(hits["customer_id"], hits["article_id"], hits["rank"], hits["repurchase"])
    )
    assert got == [
        ("u1", 1, 2, True),
        ("u1", 5, 4, False),
        ("u2", 3, 1, True),
        ("u2", 6, 3, False),
    ]
    assert len(hit_ranks(recs, GT, purchased, k=2)) == 2  # u1의 1, u2의 3


def test_hit_profile():
    purchased = purchased_pairs(TRAIN, list(GT))
    recs = {"u1": [9, 1, 1, 5], "u2": [3, 7, 6], "u3": [8]}
    out = hit_profile(hit_ranks(recs, GT, purchased, k=4))
    assert out["users_with_hit"] == 2
    assert out["n_repurchase_hits"] == 2 and out["n_new_hits"] == 2
    assert out["mean_rank_repurchase"] == 1.5  # (2 + 1) / 2
    assert out["mean_rank_new"] == 3.5  # (4 + 3) / 2
    empty = hit_profile(hit_ranks({"u3": [8]}, GT, purchased, k=4))
    assert empty["users_with_hit"] == 0 and empty["mean_rank_new"] is None


def test_per_user_matches_mean():
    recs = {"u1": [1, 9], "u2": [6, 3], "u3": [7]}
    per_user = evaluate_users_per_user(recs, GT, 2, ["recall", "map"])
    assert per_user["recall"].tolist() == [0.5, 1.0, 0.0]
    means = evaluate_users(recs, GT, 2, ["recall", "map"])
    assert means["recall"] == pytest.approx(per_user["recall"].mean())
    assert means["map"] == pytest.approx(per_user["map"].mean())


def test_summarize_seeds():
    runs = [{"seed": 0, "recall": 0.1}, {"seed": 1, "recall": 0.3}]
    out = summarize_seeds(runs, ["recall"])
    assert out["mean"]["recall"] == pytest.approx(0.2)
    assert out["std"]["recall"] == pytest.approx(np.std([0.1, 0.3], ddof=1))
    assert summarize_seeds(runs[:1], ["recall"])["std"]["recall"] == 0
