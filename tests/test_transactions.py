"""transactions 정제 테스트.

week_idx 경계(마지막 날 = 0, 7일 전 = 1)와 dtype, article 필터,
customers에 없는 customer_id 개수 세기를 가짜 데이터로 확인한다.
"""

import pandas as pd
from src.data.transactions import (
    add_week_idx,
    count_unknown_customers,
    filter_valid_articles,
)


def make_tx():
    return pd.DataFrame(
        {
            "t_dat": pd.to_datetime(
                ["2020-09-22", "2020-09-16", "2020-09-15", "2020-09-08"]
            ),
            "customer_id": pd.Series(["a", "b", "a", "z"], dtype="category"),
            "article_id": pd.Series([1, 2, 3, 1], dtype="int32"),
        }
    )


def test_week_idx_boundaries():
    out = add_week_idx(make_tx(), pd.Timestamp("2020-09-22"))
    # 마지막 날 = 0, 6일 전 = 0, 7일 전 = 1, 14일 전 = 2
    assert out["week_idx"].tolist() == [0, 0, 1, 2]
    assert out["week_idx"].dtype == "int16"


def test_filter_valid_articles():
    out = filter_valid_articles(make_tx(), pd.Series([1, 3]))
    assert out["article_id"].tolist() == [1, 3, 1]


def test_count_unknown_customers():
    assert count_unknown_customers(make_tx(), pd.Series(["a", "b"])) == 1
