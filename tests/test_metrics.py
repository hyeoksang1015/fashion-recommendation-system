"""평가 지표와 정답셋 테스트.

손으로 계산 가능한 예시(추천 5개, 정답 3개)로 precision/recall/nDCG/AP@k 값을 검증하고,
정답셋에 없는 유저, 빈 추천, 전부 맞음/전부 틀림, 중복 추천, 추천/정답 타입 불일치
edge case를 확인한다. article_id는 parquet과 같은 정수를 쓴다.
"""

import math
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.evaluation.ground_truth import build_ground_truth
from src.evaluation.metrics import (
    METRIC_FUNCS,
    average_precision_at_k,
    evaluate_users,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.utils.config import load_config

REC = [101, 102, 103, 104, 105]  # hit: 1위 101, 3위 103
REL = {101, 103, 999}


def test_hand_computed_k5():
    assert precision_at_k(REC, REL, 5) == pytest.approx(2 / 5)
    assert recall_at_k(REC, REL, 5) == pytest.approx(2 / 3)
    # DCG = 1/log2(2) + 1/log2(4) = 1.5, IDCG = 1 + 1/log2(3) + 1/log2(4)
    idcg = 1 + 1 / math.log2(3) + 0.5
    assert ndcg_at_k(REC, REL, 5) == pytest.approx(1.5 / idcg)
    # AP = (1/1 + 2/3) / min(3, 5)
    assert average_precision_at_k(REC, REL, 5) == pytest.approx((1 + 2 / 3) / 3)


def test_hand_computed_k2():
    assert precision_at_k(REC, REL, 2) == pytest.approx(1 / 2)
    assert recall_at_k(REC, REL, 2) == pytest.approx(1 / 3)
    assert ndcg_at_k(REC, REL, 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    # AP 분모 = min(3, 2) = 2
    assert average_precision_at_k(REC, REL, 2) == pytest.approx(1 / 2)


def test_all_hit():
    rel = {1, 2, 3}
    rec = [1, 2, 3]
    assert precision_at_k(rec, rel, 3) == 1
    assert recall_at_k(rec, rel, 3) == 1
    assert ndcg_at_k(rec, rel, 3) == pytest.approx(1)
    assert average_precision_at_k(rec, rel, 3) == 1


def test_all_miss_and_empty():
    for rec in ([7, 8, 9], []):
        for func in METRIC_FUNCS.values():
            assert func(rec, REL, 5) == 0


def test_duplicate_recommendation_counted_once():
    assert precision_at_k([1, 1, 1], {1}, 3) == pytest.approx(1 / 3)
    assert average_precision_at_k([1, 1], {1}, 2) == 1


def test_empty_relevant_raises():
    with pytest.raises(ValueError):
        recall_at_k(REC, set(), 5)


def test_evaluate_users():
    gt = {"u1": REL, "u2": {555}}
    recs = {"u1": REC, "u3": [555]}  # u2 추천 없음 -> 0점, u3 정답셋에 없음 -> 제외
    out = evaluate_users(recs, gt, 5, ["precision", "map"])
    assert out["precision"] == pytest.approx((2 / 5 + 0) / 2)
    assert out["map"] == pytest.approx(((1 + 2 / 3) / 3 + 0) / 2)


def test_evaluate_users_numpy_int_recommendations():
    # 모델이 numpy int32로 추천해도 파이썬 int 정답과 같은 타입으로 본다
    recs = {"u1": np.array(REC, dtype="int32").tolist()}
    recs_np = {"u1": list(np.array(REC, dtype="int32"))}
    gt = {"u1": REL}
    expected = evaluate_users(recs, gt, 5, ["map"])
    assert evaluate_users(recs_np, gt, 5, ["map"]) == expected


def test_evaluate_users_type_mismatch_raises():
    gt = {"u1": REL}
    recs = {"u1": [str(i) for i in REC]}
    with pytest.raises(ValueError, match=r"dtype\(str\).*dtype\(int\)"):
        evaluate_users(recs, gt, 5, ["map"])


def test_evaluate_users_unknown_metric():
    with pytest.raises(ValueError):
        evaluate_users({}, {"u1": {1}}, 5, ["auc"])


def test_build_ground_truth():
    df = pd.DataFrame(
        {
            "customer_id": pd.Series(["u1", "u1", "u2", "u1"], dtype="category"),
            "article_id": pd.Series([108775015, 2, 3, 108775015], dtype="int32"),
        }
    )
    gt = build_ground_truth(df)
    assert gt == {"u1": {108775015, 2}, "u2": {3}}
    assert all(isinstance(i, int) for s in gt.values() for i in s)


def test_evaluation_config():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "evaluation.yaml")
    assert cfg["k"] == 12
    assert set(cfg["metrics"]) <= set(METRIC_FUNCS)
