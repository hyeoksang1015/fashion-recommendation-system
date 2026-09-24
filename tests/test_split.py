"""시간 분할 테스트.

train / valid / test 구간이 겹치지 않고 행 수 합이 전체와 같은지,
여러 주로 나눌 때도 경계가 맞는지 가짜 week_idx로 확인한다.
"""

import pandas as pd
from src.data.split import split_by_week


def test_split_disjoint_and_complete():
    df = pd.DataFrame({"week_idx": [0, 0, 1, 1, 2, 3, 5], "row": range(7)})
    train, valid, test = split_by_week(df, test_weeks=1, valid_weeks=1)
    assert test["week_idx"].tolist() == [0, 0]
    assert valid["week_idx"].tolist() == [1, 1]
    assert train["week_idx"].tolist() == [2, 3, 5]
    idx = [set(p.index) for p in (train, valid, test)]
    assert not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])
    assert len(train) + len(valid) + len(test) == len(df)


def test_split_multi_week():
    df = pd.DataFrame({"week_idx": list(range(6))})
    train, valid, test = split_by_week(df, test_weeks=2, valid_weeks=2)
    assert test["week_idx"].tolist() == [0, 1]
    assert valid["week_idx"].tolist() == [2, 3]
    assert train["week_idx"].tolist() == [4, 5]
