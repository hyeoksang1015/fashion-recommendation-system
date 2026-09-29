"""ALS 조합 실험 CLI 진입점.

역할: configs/als_experiments.yaml의 조합을 als.yaml 기본값 위에 덮어써 실행하고
    결과를 비교한다.
동작: run_experiments로 조합별 전체 지표, 커버율, 추천 상품 종류 수, 재구매 추천
    비율을 output_path json에 저장하고, 조합별 요약 표를 로그로 낸다.
사용법: python scripts/run_als_experiments.py --config configs/als_experiments.yaml
"""

import argparse
import logging
from src.pipeline.als import run_experiments
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def format_summary(records: list[dict]) -> str:
    """조합별 recall/MAP/nDCG 요약 표를 만든다.

    Args:
        records: run_experiments 결과.

    Returns:
        로그로 낼 표 문자열.
    """
    lines = [
        f"{'params':<52}{'recall':>9}{'map':>9}{'ndcg':>9}"
        f"{'coverage':>10}{'items':>8}{'repurch':>9}"
    ]
    for r in records:
        params = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        s = r["all"]
        lines.append(
            f"{params:<52}{s['recall']:>9.4f}{s['map']:>9.4f}{s['ndcg']:>9.4f}"
            f"{r['coverage']:>10.2%}{r['n_distinct_items']:>8}"
            f"{r['repurchase_rate']:>9.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    """실험 config 경로를 받아 ALS 조합 실험을 실행한다."""
    parser = argparse.ArgumentParser(description="ALS 조합 실험")
    parser.add_argument("--config", default="configs/als_experiments.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    records = run_experiments(load_config(args.config))
    logger.info("\n%s", format_summary(records))


if __name__ == "__main__":
    main()
