"""week_idx 기준 train / valid / test 시간 분할.

역할: 모든 모델이 같은 기간으로 학습/평가하도록 거래를 세 구간으로 나눈다.
동작: week_idx가 작을수록 최근이다. test = 최근 test_weeks주, valid = 그 이전
    valid_weeks주, train = 나머지. 불리언 마스크로 나누므로 구간이 겹치지 않는다.
"""

import pandas as pd


def split_by_week(
    df: pd.DataFrame, test_weeks: int, valid_weeks: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """week_idx로 train / valid / test를 나눈다.

    test: week_idx < test_weeks
    valid: test_weeks <= week_idx < test_weeks + valid_weeks
    train: 나머지

    Args:
        df: week_idx가 있는 transactions DataFrame.
        test_weeks: test 주 수.
        valid_weeks: valid 주 수.

    Returns:
        (train, valid, test) 튜플.
    """
    week = df["week_idx"]
    is_test = week < test_weeks
    is_valid = ~is_test & (week < test_weeks + valid_weeks)
    is_train = ~is_test & ~is_valid
    return df[is_train], df[is_valid], df[is_test]
