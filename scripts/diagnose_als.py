"""ALS 시작값 진단 CLI 진입점.

역할: ALS와 age_fallback의 valid hit를 재구매/신규 구매로 나누고, ALS 커버 유저만
    놓고 두 모델을 같은 유저 집합에서 비교한다.
동작: --config로 받은 als config로 run_diagnosis를 실행하고 결과 표를 로그로 낸다.
사용법: python scripts/diagnose_als.py --config configs/als.yaml
"""

import argparse
import logging
from src.pipeline.als import format_hits, run_diagnosis
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def format_diagnosis(result: dict) -> str:
    """진단 결과를 표 문자열로 만든다.

    Args:
        result: run_diagnosis 결과.

    Returns:
        로그로 낼 표 문자열.
    """
    lines = ["[hit 분해: 전체 정답 유저]", format_hits(result["hits"])]

    covered = result["covered_users"]
    names = list(covered)
    header = f"{'':<22}" + "".join(f"{n:>14}" for n in names)
    n_users = covered[names[0]]["n_users"]
    lines += [
        "",
        f"[ALS 커버 유저만: {n_users}명, 커버율 {result['coverage']:.2%}]",
        header,
    ]
    metrics = [m for m in covered[names[0]] if m != "n_users"]
    for m in metrics:
        lines.append(
            f"{m + '@k':<22}" + "".join(f"{covered[n][m]:>14.4f}" for n in names)
        )
    return "\n".join(lines)


def main() -> None:
    """config 경로를 받아 ALS 진단을 실행한다."""
    parser = argparse.ArgumentParser(description="ALS 시작값 진단")
    parser.add_argument("--config", default="configs/als.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run_diagnosis(load_config(args.config))
    logger.info("\n%s", format_diagnosis(result))


if __name__ == "__main__":
    main()
