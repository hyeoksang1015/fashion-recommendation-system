"""이미지 프로필 추천 vs ALS 평가 CLI 진입점.

역할: 이미지 임베딩 프로필 추천이 ALS와 비교해 어떤지, 커버 유저 기준으로 평가한다.
동작:
    1. run_image_profile로 ALS 학습, 이미지 프로필 추천, 평가를 수행한다.
    2. 지표 표(전체, 그룹별 recall)와 hit 구성(재구매 vs 신규 구매)을 로그로 낸다.
    3. 결과를 {output_dir}/image_profile_results.json에 저장한다.
사용법: python scripts/run_image_profile.py --config configs/image_profile.yaml
선행 조건: data/processed, data/embeddings(embeddings.npy, article_ids.npy)
"""

import argparse
import logging
import os
from src.pipeline.als import format_comparison, format_hits
from src.pipeline.baseline import save_results
from src.pipeline.image_profile import run_image_profile
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    """config 경로를 받아 이미지 프로필 추천 평가를 실행한다."""
    parser = argparse.ArgumentParser(description="이미지 프로필 추천 평가")
    parser.add_argument("--config", default="configs/image_profile.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    out = run_image_profile(config)
    logger.info("\n%s", format_comparison(out["results"], out["metrics"]))
    logger.info("\n%s", format_hits(out["diagnostics"]))
    save_results(
        os.path.join(config["output_dir"], "image_profile_results.json"),
        out["split"],
        out["k"],
        out["results"],
        diagnostics=out["diagnostics"],
        info=out["info"],
    )


if __name__ == "__main__":
    main()
