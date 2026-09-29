"""age_fallback 인기 집계 기간 비교 CLI 진입점.

역할: 나이대별 인기 집계 기간(configs/features.yaml의 age_group_week_window, 현재 4주)
    이 적절한지 기간 후보별 valid 지표로 비교한다.
동작: run_age_window_check로 기간마다 age_fallback 추천을 만들어 평가하고,
    output_path json에 저장한 뒤 기간별 표를 로그로 낸다.
사용법: python scripts/run_age_window_check.py --config configs/age_window_check.yaml
"""

import argparse
import logging
from src.pipeline.baseline import run_age_window_check
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def format_windows(results: dict) -> str:
    """기간별 지표 표를 만든다.

    Args:
        results: run_age_window_check 결과.

    Returns:
        로그로 낼 표 문자열.
    """
    names = list(results)
    metrics = [m for m in results[names[0]] if m != "n_users"]
    lines = [f"{'':<14}" + "".join(f"{m + '@k':>14}" for m in metrics)]
    for name in names:
        lines.append(
            f"{name:<14}" + "".join(f"{results[name][m]:>14.4f}" for m in metrics)
        )
    return "\n".join(lines)


def main() -> None:
    """config 경로를 받아 기간 비교를 실행한다."""
    parser = argparse.ArgumentParser(description="age_fallback 인기 기간 비교")
    parser.add_argument("--config", default="configs/age_window_check.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results = run_age_window_check(load_config(args.config))
    logger.info("\n%s", format_windows(results))


if __name__ == "__main__":
    main()
