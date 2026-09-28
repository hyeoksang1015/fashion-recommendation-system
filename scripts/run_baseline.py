"""인기 baseline 평가 CLI 진입점.

역할: 커맨드라인에서 baseline 3종(overall / recent / age_fallback) 평가를 실행한다.
동작: --config로 받은 yaml을 읽고 logging을 설정한 뒤 run_baseline을 호출한다.
    결과는 config의 output_dir/baseline_results.json에 저장된다.
사용법: python scripts/run_baseline.py --config configs/baseline.yaml
"""

import argparse
import logging
from src.pipeline.baseline import run_baseline
from src.utils.config import load_config


def main() -> None:
    """config 경로를 받아 baseline 평가를 실행한다."""
    parser = argparse.ArgumentParser(description="인기 baseline 평가")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_baseline(load_config(args.config))


if __name__ == "__main__":
    main()
