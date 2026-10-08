"""이미지 프로필 추천 평가 파이프라인.

역할: 유저가 산 상품의 이미지 임베딩으로 만든 프로필 추천을, 같은 유저에 대한
    ALS 추천과 같은 평가 환경(정답셋, 지표, 유저/상품 그룹)으로 비교한다.
동작:
    1. ALS와 같은 config(als_best)로 평가 환경을 만들고 ALS를 학습한다.
    2. ALS 행렬에 이력이 있는 커버 유저만 평가 대상으로 삼는다 (프로필에 이력이 필요).
    3. 커버 유저에게 이미지 프로필 추천(후보는 임베딩 전체)과 ALS 추천을 만든다.
       이미 산 상품을 뺀 변형(als_filter, image_filter)도 함께 만든다.
    4. 같은 정답셋으로 추천을 평가하고, 재구매 비율과 hit 구성을 진단한다.
"""

import logging
from src.evaluation.diagnostics import (
    decompose_hits,
    n_distinct_items,
    purchased_pairs,
    repurchase_rate,
)
from src.features.image_profile import load_embeddings, recommend_for_users
from src.pipeline.als import (
    MODEL,
    apply_overrides,
    evaluate_models,
    fit_als,
    load_eval_context,
    recommend_with_fallback,
)
from src.utils.config import load_config

logger = logging.getLogger(__name__)

IMAGE = "image_profile"
# 이미 산 상품을 뺀 추가 비교 (7.2절). 이름은 표 열 너비(14자)에 맞춘다.
ALS_FILTER = "als_filter"
IMAGE_FILTER = "image_filter"


def diagnose(
    recs: dict[str, list[int]],
    ground_truth: dict[str, set[int]],
    purchased,
    k: int,
) -> dict:
    """추천의 hit 구성, 재구매 비율, 추천 상품 종류 수를 계산한다.

    Args:
        recs: {customer_id: 순위순 추천 리스트}.
        ground_truth: {customer_id: 정답 집합}.
        purchased: purchased_pairs 결과 (train에서 이미 산 쌍).
        k: 자를 순위.

    Returns:
        hits, repurchase_hits, new_purchase_hits, repurchase_rate,
        distinct_items 키를 가진 dict.
    """
    return {
        **decompose_hits(recs, ground_truth, purchased, k),
        "repurchase_rate": round(repurchase_rate(recs, purchased, k), 4),
        "distinct_items": n_distinct_items(recs),
    }


def run_image_profile(config: dict) -> dict:
    """이미지 프로필 추천과 ALS를 커버 유저 기준으로 평가한다.

    Args:
        config: image_profile config (als_config, embedding_dir, chunk_size,
            item_group_metrics).

    Returns:
        split, k, metrics, results, diagnostics, info 키를 가진 dict.

    Raises:
        ValueError: 어떤 모델의 추천 유저가 커버 유저와 다를 때.
    """
    als_config = apply_overrides(load_config(config["als_config"]))
    ctx = load_eval_context(als_config)
    k = als_config["k"]

    fitted = fit_als(ctx["train"], als_config)
    als_recs, _ = recommend_with_fallback(
        fitted, ctx["users"], ctx["fallback"], als_config
    )
    filtered_config = {**als_config, "filter_already_purchased": True}
    als_filtered, _ = recommend_with_fallback(
        fitted, ctx["users"], ctx["fallback"], filtered_config
    )
    users = list(als_recs)
    ground_truth = {user: ctx["ground_truth"][user] for user in users}

    article_ids, embeddings = load_embeddings(config["embedding_dir"])
    image_args = (
        fitted["matrix"],
        fitted["user_ids"],
        fitted["item_ids"],
        article_ids,
        embeddings,
        users,
        k,
        config["chunk_size"],
    )
    image_recs = recommend_for_users(*image_args)
    image_filtered = recommend_for_users(*image_args, filter_already_purchased=True)
    models = {
        MODEL: als_recs,
        IMAGE: image_recs,
        ALS_FILTER: als_filtered,
        IMAGE_FILTER: image_filtered,
    }
    for name, recs in models.items():
        if set(recs) != set(users):
            raise ValueError(
                f"{name} 추천 유저 {len(recs)}명, 커버 유저 {len(users)}명"
            )

    covered_ctx = {**ctx, "ground_truth": ground_truth, "users": users}
    results = evaluate_models(covered_ctx, models, config["item_group_metrics"])
    purchased = purchased_pairs(ctx["train"], users)
    diagnostics = {
        name: diagnose(recs, ground_truth, purchased, k)
        for name, recs in models.items()
    }
    logger.info("커버 유저 %d명 / 전체 %d명", len(users), len(ctx["users"]))
    return {
        "split": ctx["split"],
        "k": k,
        "metrics": ctx["eval_cfg"]["metrics"],
        "results": results,
        "diagnostics": diagnostics,
        "info": {"covered_users": len(users), **fitted["info"]},
    }
