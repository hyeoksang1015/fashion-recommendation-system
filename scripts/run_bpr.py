"""BPR 학습/평가 CLI 진입점.

역할: train으로 BPR을 학습하고, age_fallback, ALS(최종 설정)와 같은 평가 환경
    (정답셋, 지표, 유저/상품 그룹)에서 비교한다.
동작:
    1. load_eval_context로 전처리 parquet, 정답셋, age_fallback 추천을 만든다.
    2. ALS는 als_config(최고 조합)로, BPR은 bpr config로 fit_als / fit_bpr로 학습한다.
       두 모델 모두 추천을 못 준 유저는 age_fallback 목록으로 채운다.
    3. evaluate_models로 세 모델을 같은 그룹 구조로 평가한다.
    4. BPR 결과를 {output_dir}/bpr_results.json에 저장하고 비교 표를 로그로 낸다.
사용법: python scripts/run_bpr.py --config configs/bpr.yaml
"""

import argparse
import logging
import os
from src.pipeline.als import (
    BASELINE,
    MODEL,
    apply_overrides,
    evaluate_models,
    fit_als,
    fit_bpr,
    format_comparison,
    load_eval_context,
    recommend_with_fallback,
)
from src.pipeline.baseline import save_results
from src.utils.config import load_config

logger = logging.getLogger(__name__)

BPR = "bpr"
PARAM_KEYS = (
    "factors",
    "learning_rate",
    "regularization",
    "iterations",
    "train_weeks",
    "filter_already_purchased",
    "k",
    "seed",
    "num_threads",
)


def main() -> None:
    """config 경로를 받아 BPR 학습과 세 모델 비교 평가를 실행한다."""
    parser = argparse.ArgumentParser(description="BPR 학습/평가")
    parser.add_argument("--config", default="configs/bpr.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    als_config = apply_overrides(load_config(config["als_config"]))
    ctx = load_eval_context(config)
    k, metrics = ctx["eval_cfg"]["k"], ctx["eval_cfg"]["metrics"]
    users, fallback = ctx["users"], ctx["fallback"]

    fitted = {
        MODEL: (fit_als(ctx["train"], als_config), als_config),
        BPR: (fit_bpr(ctx["train"], config), config),
    }
    recs = {BASELINE: fallback}
    info = {}
    for name, (fit, cfg) in fitted.items():
        model_recs, recs[name] = recommend_with_fallback(fit, users, fallback, cfg)
        info[name] = {
            **fit["info"],
            "covered_users": len(model_recs),
            "coverage": len(model_recs) / len(users),
        }
        logger.info(
            "%s: 행렬 %d x %d, nnz %d | 행렬 생성 %.1fs, 학습 %.1fs | "
            "커버 %d / %d명 (%.2f%%)",
            name,
            info[name]["n_users"],
            info[name]["n_items"],
            info[name]["nnz"],
            info[name]["build_matrix_seconds"],
            info[name]["train_seconds"],
            len(model_recs),
            len(users),
            100 * info[name]["coverage"],
        )
    del fitted

    results = evaluate_models(ctx, recs, config["item_group_metrics"])
    logger.info("\n%s", format_comparison(results, metrics))
    save_results(
        os.path.join(config["output_dir"], "bpr_results.json"),
        ctx["split"],
        k,
        {BPR: results[BPR]},
        bpr_info={**info[BPR], "params": {key: config[key] for key in PARAM_KEYS}},
    )


if __name__ == "__main__":
    main()
