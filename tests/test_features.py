"""features 모듈 테스트.

작은 가짜 train(10행)으로 인기도(distinct 구매자 수), fallback 테이블, 평가 그룹 라벨을
손으로 계산한 값과 비교한다. edge case: 구매 이력이 없는 age_group, train에 없던 신규
article, cold_threshold 경계값(정확히 threshold회).
"""

import pandas as pd
import pytest
from pathlib import Path
from src.features.fallback import build_fallback_table
from src.features.groups import (
    label_age_group,
    label_item_frequency,
    label_new_items,
    label_unknown_metadata,
    label_user_activity,
)
from src.features.popularity import (
    compute_age_group_popularity,
    compute_overall_popularity,
    compute_recent_popularity,
)
from src.utils.config import load_config

AGE_LABELS = ["20s", "30s", "40s", "Unknown"]


def make_train():
    # train의 최소 week_idx는 2 (0=test, 1=valid)
    rows = [
        # customer, article, week_idx
        ("a", 1, 2),
        ("a", 1, 2),  # a의 재구매: 구매자 수에는 1명으로만 센다
        ("a", 1, 3),
        ("b", 1, 2),
        ("b", 2, 2),
        ("c", 2, 3),
        ("c", 3, 5),
        ("d", 3, 6),
        ("d", 4, 7),
        ("e", 3, 9),
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "article_id", "week_idx"])
    return df.astype(
        {"customer_id": "category", "article_id": "int32", "week_idx": "int16"}
    )


def make_customers():
    df = pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d", "e"],
            "age_group": ["20s", "20s", "30s", "Unknown", "30s"],
        }
    )
    # 40s는 구매 이력이 없는 그룹
    df["age_group"] = pd.Categorical(df["age_group"], categories=AGE_LABELS)
    return df


def as_pairs(df, cols=("article_id", "n_buyers")):
    return list(df[list(cols)].itertuples(index=False, name=None))


def test_overall_popularity_counts_distinct_buyers():
    # 1: a,b / 2: b,c / 3: c,d,e / 4: d
    out = compute_overall_popularity(make_train())
    assert as_pairs(out) == [(3, 3), (1, 2), (2, 2), (4, 1)]  # 동률은 article_id 순


def test_recent_popularity_last_week():
    # 마지막 1주 = week_idx 2: 1: a,b / 2: b
    out = compute_recent_popularity(make_train(), n_weeks=1)
    assert as_pairs(out) == [(1, 2), (2, 1)]
    # 마지막 2주 = week_idx 2~3: 1: a,b / 2: b,c
    out = compute_recent_popularity(make_train(), n_weeks=2)
    assert as_pairs(out) == [(1, 2), (2, 2)]


def test_age_group_popularity_last_4_weeks():
    # week_idx 2~5: 20s(a,b): 1 -> a,b / 2 -> b ; 30s(c): 2, 3
    out = compute_age_group_popularity(make_train(), make_customers(), n_weeks=4)
    assert as_pairs(out, ("age_group", "article_id", "n_buyers")) == [
        ("20s", 1, 2),
        ("20s", 2, 1),
        ("30s", 2, 1),
        ("30s", 3, 1),
    ]
    # 구매가 없는 그룹도 category에는 남는다
    assert list(out["age_group"].cat.categories) == AGE_LABELS


def test_fallback_table():
    train = make_train()
    overall = compute_overall_popularity(train)  # [3, 1, 2, 4]
    by_age = compute_age_group_popularity(train, make_customers(), n_weeks=4)
    table = build_fallback_table(overall, by_age, k=3, unknown_label="Unknown")
    assert table == {
        "20s": [1, 2, 3],  # 그룹 2개 + 전체 인기로 채움
        "30s": [2, 3, 1],
        "40s": [3, 1, 2],  # 구매 이력 없음 -> 전체 인기
        "Unknown": [3, 1, 2],  # Unknown은 항상 전체 인기
    }
    assert all(isinstance(a, int) for items in table.values() for a in items)


def test_user_activity_threshold_boundary():
    rows = [("x", 1, 2)] * 5 + [("y", 1, 2)] * 6 + [("z", 1, 2)]
    train = pd.DataFrame(rows, columns=["customer_id", "article_id", "week_idx"])
    out = label_user_activity(train, cold_threshold=5)
    assert out == {"x": "cold", "y": "warm", "z": "cold"}  # 정확히 5회는 cold


def test_item_frequency_terciles():
    # 상품 a의 구매자 수 = a명 (1~6). 분위 경계가 겹치지 않게 모두 다른 값으로 만든다
    rows = [
        (f"u{i}", a, 2)
        for a, n in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
        for i in range(n)
    ]
    train = pd.DataFrame(rows, columns=["customer_id", "article_id", "week_idx"])
    out = label_item_frequency(train, quantiles=[0.33, 0.67])
    assert out == {1: "low", 2: "low", 3: "mid", 4: "mid", 5: "high", 6: "high"}


def test_new_items():
    target = pd.DataFrame({"article_id": pd.Series([1, 99, 3, 99], dtype="int32")})
    assert label_new_items(make_train(), target) == {1: False, 99: True, 3: False}


def test_unknown_metadata():
    articles = pd.DataFrame(
        {"article_id": [1, 2], "product_group_name": ["Unknown", "Shoes"]}
    )
    assert label_unknown_metadata(articles, "Unknown") == {1: True, 2: False}


def test_age_group_dict():
    assert label_age_group(make_customers()) == {
        "a": "20s",
        "b": "20s",
        "c": "30s",
        "d": "Unknown",
        "e": "30s",
    }


def test_features_config():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "features.yaml")
    assert cfg["recent_week_window"] == 1
    assert cfg["age_group_week_window"] == 4
    assert cfg["top_k"] == 12
    assert cfg["cold_warm_threshold"] == 5
    assert cfg["item_frequency_quantiles"] == [0.33, 0.67]


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        compute_recent_popularity(make_train(), n_weeks=0)
