"""ALS 최고 조합 검증 CLI 진입점.

역할: configs/als_best.yaml의 최고 조합을 baseline과 같은 그룹 구조 + ALS 커버/미커버
    그룹으로 평가하고, hit를 재구매/신규로 나누고, 평가 주를 과거로 옮긴 분할에서
    age_fallback과 다시 비교한다.
동작: run_best_validation 결과를 output_path json에 저장하고 표를 로그로 낸다.
사용법: python scripts/validate_als_best.py --config configs/als_best.yaml
"""

import argparse
import logging
from src.pipeline.als import format_comparison, format_hits, run_best_validation
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def format_recheck(recheck: dict) -> str:
    """재확인 평가의 all 지표와 커버율을 표 문자열로 만든다.

    Args:
        recheck: run_best_validation 결과의 recheck.

    Returns:
        로그로 낼 표 문자열.
    """
    results = recheck["results"]
    names = list(results)
    metrics = [m for m in results[names[0]] if m != "n_users"]
    lines = [
        f"[재확인: 평가 week_idx {recheck['target_week_idx']}, "
        f"train week_idx >= {recheck['train_week_idx_min']}, "
        f"유저 {results[names[0]]['n_users']}명, "
        f"ALS 커버율 {recheck['coverage']:.2%}]",
        f"{'':<22}" + "".join(f"{n:>14}" for n in names),
    ]
    for m in metrics:
        lines.append(
            f"{m + '@k':<22}" + "".join(f"{results[n][m]:>14.4f}" for n in names)
        )
    return "\n".join(lines)


def main() -> None:
    """검증 config 경로를 받아 최고 조합 검증을 실행한다."""
    parser = argparse.ArgumentParser(description="ALS 최고 조합 검증")
    parser.add_argument("--config", default="configs/als_best.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    out = run_best_validation(load_config(args.config))
    metrics = [m for m in out["results"]["als"]["all"] if m != "n_users"]
    logger.info(
        "\nALS 커버율 %.2f%%, 학습 %s\n%s\n\n[hit 분해]\n%s\n\n%s",
        100 * out["als_info"]["coverage"],
        out["als_info"]["fit_info"],
        format_comparison(out["results"], metrics),
        format_hits(out["hits"]),
        format_recheck(out["recheck"]),
    )


if __name__ == "__main__":
    main()
