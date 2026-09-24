"""customers 정제 테스트.

구독 OR 규칙(대소문자/공백 정규화 포함), Active/club_member_status 결측 처리,
age 경계값(19, 20, 29, 30)과 결측/-1 -> Unknown, 행 수 유지를 가짜 데이터로
확인한다.
"""

import numpy as np
import pandas as pd
from src.data.customers import (
    add_age_group,
    add_fashion_news_subscribed,
    fill_active,
    fill_club_member_status,
)

BINS = [0, 19, 29, 39, 49, 59, 120]
LABELS = ["16-19", "20s", "30s", "40s", "50s", "60+"]


def make_customers():
    return pd.DataFrame(
        {
            "FN": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            "fashion_news_frequency": [
                "NONE",
                "Regularly",
                " monthly",
                "None",
                None,
                "NONE",
            ],
            "Active": [1.0, np.nan, 1.0, np.nan, np.nan, np.nan],
            "club_member_status": [
                "ACTIVE",
                None,
                "LEFT CLUB",
                None,
                "PRE-CREATE",
                None,
            ],
            "age": [19, 20, 29, 30, np.nan, -1],
        }
    )


def test_subscribed_or_rule():
    out = add_fashion_news_subscribed(make_customers(), ["Regularly", "Monthly"])
    assert out["fashion_news_subscribed"].tolist() == [1, 1, 1, 0, 0, 0]
    assert out["fashion_news_subscribed"].dtype == "int8"


def test_active_and_club_status():
    out = fill_club_member_status(fill_active(make_customers()), "Unknown")
    assert out["Active"].tolist() == [1, 0, 1, 0, 0, 0]
    assert out["club_member_status"].isna().sum() == 0


def test_age_group_boundaries_and_unknown():
    df = make_customers()
    out = add_age_group(df, BINS, LABELS, "Unknown")
    assert out["age_group"].astype(str).tolist() == [
        "16-19",
        "20s",
        "20s",
        "30s",
        "Unknown",
        "Unknown",  # -1 -> NaN -> Unknown
    ]
    assert np.isnan(out.loc[5, "age"])


def test_row_count_preserved():
    df = make_customers()
    out = add_age_group(
        fill_club_member_status(
            fill_active(add_fashion_news_subscribed(df, ["Regularly"])), "Unknown"
        ),
        BINS,
        LABELS,
        "Unknown",
    )
    assert len(out) == len(df)
