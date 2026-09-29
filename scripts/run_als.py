"""ALS 학습/평가 CLI 진입점.

역할: train으로 ALS를 학습하고, 평가 분할(기본 valid)에서 baseline(age_fallback)과
    같은 평가 환경(정답셋, 지표, 유저/상품 그룹)으로 비교한다.
동작:
    1. load_eval_context로 전처리 parquet, 정답셋, age_fallback 추천을 만든다.
    2. fit_als로 학습하고, ALS가 추천을 준 유저는 ALS 추천으로 덮어쓴다.
       즉 ALS에 없는 유저는 age_fallback 목록(나이 결측/customers에 없으면
       unknown_label 목록)을 받는다.
    3. evaluate_models(baseline의 그룹 평가 재사용)로 age_fallback과 ALS를
       같은 그룹 구조로 평가한다.
    4. ALS 결과를 {output_dir}/als_results.json에 저장하고 비교 표를 로그로 낸다.
사용법: python scripts/run_als.py --config configs/als.yaml
"""

import argparse
import logging
import os
from src.pipeline.als import (
    BASELINE,
    MODEL,
    evaluate_models,
    fit_als,
    format_comparison,
    load_eval_context,
    recommend_with_fallback,
)
from src.pipeline.baseline import save_results
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    """config 경로를 받아 ALS 학습과 평가를 실행한다."""
    parser = argparse.ArgumentParser(description="ALS 학습/평가")
    parser.add_argument("--config", default="configs/als.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    ctx = load_eval_context(config)
    k, metrics = ctx["eval_cfg"]["k"], ctx["eval_cfg"]["metrics"]
    users, fallback = ctx["users"], ctx["fallback"]

    fitted = fit_als(ctx["train"], config)
    als_recs, recs = recommend_with_fallback(fitted, users, fallback, config)

    results = evaluate_models(
        ctx, {BASELINE: fallback, MODEL: recs}, config["item_group_metrics"]
    )

    info = {
        **fitted["info"],
        "als_covered_users": len(als_recs),
        "als_coverage": len(als_recs) / len(users),
        "params": {
            key: config[key]
            for key in (
                "factors",
                "regularization",
                "iterations",
                "alpha",
                "train_weeks",
                "filter_already_purchased",
                "k",
                "seed",
            )
        },
    }
    logger.info(
        "행렬 %d x %d, nnz %d | 행렬 생성 %.1fs, 학습 %.1fs | "
        "ALS 커버 %d / %d명 (%.2f%%)",
        info["n_users"],
        info["n_items"],
        info["nnz"],
        info["build_matrix_seconds"],
        info["train_seconds"],
        len(als_recs),
        len(users),
        100 * info["als_coverage"],
    )
    logger.info("\n%s", format_comparison(results, metrics))
    save_results(
        os.path.join(config["output_dir"], "als_results.json"),
        ctx["split"],
        k,
        {MODEL: results[MODEL]},
        als_info=info,
    )


if __name__ == "__main__":
    main()
