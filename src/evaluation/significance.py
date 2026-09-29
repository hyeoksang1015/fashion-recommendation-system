"""두 모델의 유저별 점수 차이 검정.

역할: 같은 유저 집합에서 두 모델의 유저별 점수로 평균 차이와 신뢰구간을 낸다.
동작: paired bootstrap. 유저 인덱스를 (B, n) 행렬로 한 번에 복원 추출하고,
    유저별 차이 배열을 그 인덱스로 모아 행 평균을 낸다. 신뢰구간은 B개 재표본
    평균 차이의 백분위수다.
"""

import numpy as np


def paired_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int,
    seed: int,
    ci_level: float,
) -> dict[str, float]:
    """b - a 평균 차이와 paired bootstrap 신뢰구간을 계산한다.

    Args:
        scores_a: 모델 a의 유저별 점수 (길이 n).
        scores_b: 모델 b의 유저별 점수 (a와 같은 유저 순서).
        n_resamples: 재표본 수 B.
        seed: 난수 seed.
        ci_level: 신뢰수준 (예: 0.95).

    Returns:
        {"mean_diff": 원표본 평균 차이 (b - a), "ci_low", "ci_high": 신뢰구간,
        "n_users": n, "n_resamples": B}.

    Raises:
        ValueError: 두 배열 길이가 다르거나 비었을 때, ci_level이 (0, 1) 밖일 때.
    """
    if len(scores_a) != len(scores_b) or len(scores_a) == 0:
        raise ValueError(
            f"두 점수 배열의 길이가 같고 0보다 커야 함: {len(scores_a)}, "
            f"{len(scores_b)}"
        )
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level은 (0, 1) 사이여야 함: {ci_level}")
    diff = np.asarray(scores_b, dtype=np.float64) - np.asarray(
        scores_a, dtype=np.float64
    )
    n = len(diff)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n, size=(n_resamples, n), dtype=np.int32)
    resampled = diff[index].mean(axis=1)
    tail = (1 - ci_level) / 2 * 100
    low, high = np.percentile(resampled, [tail, 100 - tail])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_users": n,
        "n_resamples": n_resamples,
    }
