"""customers(유저) 정제 함수 모음.

역할: 유저 속성을 모델에서 쓸 수 있는 형태로 정리한다. 유저는 한 명도 삭제하지
    않는다(삭제하면 그 유저의 거래도 학습/평가에서 빠지기 때문).
동작: 결측은 Unknown이나 0으로 채운다. 뉴스 구독은 FN과 fashion_news_frequency를
    OR로 합치고, age는 pd.cut으로 age_group 구간을 만든다. 각 함수는 DataFrame을
    받아 새 DataFrame을 반환한다.
"""

import pandas as pd


def add_fashion_news_subscribed(
    df: pd.DataFrame, subscribed_frequencies: list[str]
) -> pd.DataFrame:
    """FN과 fashion_news_frequency를 OR로 합쳐 fashion_news_subscribed(0/1)를 만든다.

    Args:
        df: customers DataFrame.
        subscribed_frequencies: 구독으로 보는 frequency 값 목록.

    Returns:
        fashion_news_subscribed(int8) 컬럼이 추가된 DataFrame.
    """
    freq = df["fashion_news_frequency"].str.strip().str.lower()
    subscribed = {f.strip().lower() for f in subscribed_frequencies}
    out = df.copy()
    out["fashion_news_subscribed"] = ((df["FN"] == 1) | freq.isin(subscribed)).astype(
        "int8"
    )
    return out


def fill_active(df: pd.DataFrame) -> pd.DataFrame:
    """Active NaN을 비활성(0)으로 보고 int8로 바꾼다.

    Args:
        df: customers DataFrame.

    Returns:
        Active가 int8 0/1인 DataFrame.
    """
    out = df.copy()
    out["Active"] = out["Active"].fillna(0).astype("int8")
    return out


def fill_club_member_status(df: pd.DataFrame, unknown_label: str) -> pd.DataFrame:
    """club_member_status 결측을 unknown_label로 채운다.

    Args:
        df: customers DataFrame.
        unknown_label: 결측 대체 라벨.

    Returns:
        club_member_status 결측이 없는 DataFrame.
    """
    out = df.copy()
    out["club_member_status"] = out["club_member_status"].fillna(unknown_label)
    return out


def add_age_group(
    df: pd.DataFrame, bins: list[int], labels: list[str], unknown_label: str
) -> pd.DataFrame:
    """age의 -1을 NaN으로 되돌리고 age_group을 만든다. 결측은 unknown_label.

    pd.cut은 오른쪽 경계를 포함한다: (19, 29] -> 20s.

    Args:
        df: customers DataFrame.
        bins: 나이 구간 경계.
        labels: 구간 라벨.
        unknown_label: 결측 나이 라벨.

    Returns:
        age(float)와 age_group(category)이 정리된 DataFrame.
    """
    out = df.copy()
    out["age"] = out["age"].mask(out["age"] == -1).astype("float32")
    age_group = pd.cut(out["age"], bins=bins, labels=labels)
    out["age_group"] = age_group.cat.add_categories(unknown_label).fillna(unknown_label)
    return out


def drop_customer_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """분석에 쓰지 않는 컬럼을 제거한다.

    Args:
        df: customers DataFrame.
        columns: 제거할 컬럼 목록.

    Returns:
        컬럼이 제거된 DataFrame.
    """
    return df.drop(columns=columns)
