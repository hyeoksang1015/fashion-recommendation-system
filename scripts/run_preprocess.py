"""전처리 CLI 진입점.

역할: 커맨드라인에서 전처리 파이프라인을 한 번에 실행한다.
동작: --config로 받은 yaml을 읽고 logging을 설정한 뒤 run_preprocess를 호출한다.
    src는 `pip install -e .`로 설치되어 있어 sys.path 조작 없이 import된다.
사용법: python scripts/run_preprocess.py --config configs/preprocess.yaml
"""

import argparse
import logging
from src.pipeline.preprocess import run_preprocess
from src.utils.config import load_config


def main() -> None:
    """config 경로를 받아 전처리를 실행한다."""
    parser = argparse.ArgumentParser(description="H&M 전처리 파이프라인")
    parser.add_argument("--config", default="configs/preprocess.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_preprocess(load_config(args.config))


if __name__ == "__main__":
    main()
