"""ALS / BPR 차이의 견고성 확인 파이프라인.

역할: ALS와 BPR의 차이가 노이즈인지(paired bootstrap, 시드 반복) 확인하고, BPR 결과
    해석의 추측(hit가 더 많은 유저에게 흩어진다, 재구매 hit가 상위에 온다, 추천이
    다양하다)을 수치로 확인한다.
동작:
    1. als.py의 load_eval_context / fit_als / fit_bpr / recommend_with_fallback으로
       seeds마다 두 모델을 학습하고 fallback을 채운 추천의 평균 지표를 낸다.
    2. seeds[0] 모델로 세 모델(age_fallback, ALS, BPR)의 유저별 지표를 만들고,
       전체 / ALS 커버 유저(split_by_coverage)로 평균을 낸다.
    3. 같은 유저 순서의 유저별 지표로 BPR - ALS paired bootstrap을 한다.
    4. ALS / BPR 추천만(커버 유저)으로 hit 유저 수, hit 평균 순위(재구매 / 신규),
       카탈로그 커버리지를 계산한다.
"""

import logging
import numpy as np
import pandas as pd
from src.evaluation.diagnostics import hit_ranks, n_distinct_items, purchased_pairs
from src.evaluation.metrics import evaluate_users, evaluate_users_per_user
from src.evaluation.significance import paired_bootstrap
from src.pipeline.als import (
    BASELINE,
    MODEL,
    apply_overrides,
    fit_als,
    fit_bpr,
    load_eval_context,
    recommend_with_fallback,
    split_by_coverage,
)
from src.pipeline.baseline import save_results
from src.utils.config import load_config

logger = logging.getLogger(__name__)

BPR = "bpr"
ALL_USERS = "all"
COVERED_USERS = "covered"


def summarize_seeds(runs: list[dict], metrics: list[str]) -> dict:
    """시드별 지표에 평균과 표본 표준편차를 붙인다.

    Args:
        runs: [{"seed": seed, 지표: 값, ...}] 목록.
        metrics: 요약할 지표 이름.

    Returns:
        {"runs": runs, "mean": {지표: 평균}, "std": {지표: 표준편차(ddof=1)}}.
        시드가 하나면 표준편차는 0.
    """
    values = pd.DataFrame(runs)[metrics]
    return {
        "runs": runs,
        "mean": values.mean().to_dict(),
        "std": values.std(ddof=1).fillna(0.0).to_dict(),
    }


def hit_profile(hits: pd.DataFrame) -> dict:
    """hit_ranks 결과로 hit 유저 수와 재구매 / 신규 hit의 평균 순위를 낸다.

    Args:
        hits: hit_ranks 결과 (customer_id, rank, repurchase).

    Returns:
        {"hits", "users_with_hit", "mean_rank_repurchase", "mean_rank_new",
        "n_repurchase_hits", "n_new_hits"}. hit가 없는 쪽 평균 순위는 None.
    """
    ranks = hits.groupby("repurchase")["rank"]
    sizes, means = ranks.size(), ranks.mean()
    return {
        "hits": len(hits),
        "users_with_hit": int(hits["customer_id"].nunique()),
        "n_repurchase_hits": int(sizes.get(True, 0)),
        "n_new_hits": int(sizes.get(False, 0)),
        "mean_rank_repurchase": float(means[True]) if True in means else None,
        "mean_rank_new": float(means[False]) if False in means else None,
    }


def _subset_means(per_user: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    """유저별 지표 배열에서 mask 유저의 평균을 낸다.

    Args:
        per_user: {지표: 유저별 점수 배열}.
        mask: 평균에 넣을 유저 bool 배열.

    Returns:
        {지표: 평균, "n_users": 유저 수}.
    """
    return {
        **{m: float(v[mask].mean()) for m, v in per_user.items()},
        "n_users": int(mask.sum()),
    }


def run_robustness(config: dict) -> dict:
    """ALS / BPR 견고성 확인을 실행하고 json과 유저별 parquet을 저장한다.

    Args:
        config: robustness config (bpr_config, als_config, output_path,
            per_user_path, metrics, seeds, bootstrap).

    Returns:
        {"results": {all | covered: {모델: 평균 지표}}, "bootstrap", "seeds",
        "hits", "catalog"}.
    """
    bpr_cfg = load_config(config["bpr_config"])
    model_cfgs = {
        MODEL: (fit_als, apply_overrides(load_config(config["als_config"]))),
        BPR: (fit_bpr, bpr_cfg),
    }
    ctx = load_eval_context(bpr_cfg)
    users, ground_truth = ctx["users"], ctx["ground_truth"]
    k, eval_metrics = ctx["eval_cfg"]["k"], ctx["eval_cfg"]["metrics"]
    metrics, seeds = config["metrics"], config["seeds"]

    seed_runs: dict[str, list[dict]] = {name: [] for name in model_cfgs}
    models: dict[str, dict] = {}
    for seed in seeds:
        for name, (fit_fn, cfg) in model_cfgs.items():
            cfg = {**cfg, "seed": seed}
            fitted = fit_fn(ctx["train"], cfg)
            only, recs = recommend_with_fallback(fitted, users, ctx["fallback"], cfg)
            scores = evaluate_users(recs, ground_truth, k, metrics)
            seed_runs[name].append(
                {
                    "seed": seed,
                    **scores,
                    "train_seconds": fitted["info"]["train_seconds"],
                }
            )
            logger.info("%s seed=%d: %s", name, seed, scores)
            if seed == seeds[0]:
                models[name] = {
                    "only": only,
                    "recs": recs,
                    "n_items_trained": fitted["info"]["n_items"],
                }

    all_recs = {BASELINE: ctx["fallback"]} | {n: m["recs"] for n, m in models.items()}
    per_user = {
        name: evaluate_users_per_user(recs, ground_truth, k, eval_metrics)
        for name, recs in all_recs.items()
    }
    covered_gt = split_by_coverage(ground_truth, models[MODEL]["only"])[COVERED_USERS]
    masks = {
        ALL_USERS: np.ones(len(users), dtype=bool),
        COVERED_USERS: pd.Index(users).isin(list(covered_gt)),
    }
    results = {
        subset: {name: _subset_means(s, mask) for name, s in per_user.items()}
        for subset, mask in masks.items()
    }

    boot_cfg = config["bootstrap"]
    bootstrap = {
        subset: {
            m: paired_bootstrap(
                per_user[MODEL][m][mask],
                per_user[BPR][m][mask],
                boot_cfg["n_resamples"],
                boot_cfg["seed"],
                boot_cfg["ci_level"],
            )
            for m in metrics
        }
        for subset, mask in masks.items()
    }

    purchased = purchased_pairs(ctx["train"], users)
    n_catalog = len(ctx["articles"])
    hits, catalog = {}, {}
    for name, m in models.items():
        hits[name] = hit_profile(hit_ranks(m["only"], ground_truth, purchased, k))
        n_items = n_distinct_items(m["only"])
        catalog[name] = {
            "covered_users": len(m["only"]),
            "n_distinct_items": n_items,
            "n_catalog": n_catalog,
            "catalog_coverage": n_items / n_catalog,
            "n_items_trained": m["n_items_trained"],
            "trained_item_coverage": n_items / m["n_items_trained"],
        }

    per_user_df = pd.DataFrame(
        {"customer_id": users, COVERED_USERS: masks[COVERED_USERS]}
        | {f"{name}_{m}": per_user[name][m] for name in per_user for m in metrics}
    )
    per_user_df.to_parquet(config["per_user_path"], index=False)
    logger.info("저장 완료: %s", config["per_user_path"])

    out = {
        "results": results,
        "bootstrap": bootstrap,
        "seeds": {n: summarize_seeds(r, metrics) for n, r in seed_runs.items()},
        "hits": hits,
        "catalog": catalog,
    }
    save_results(
        config["output_path"],
        ctx["split"],
        k,
        results,
        bootstrap_diff="bpr - als",
        bootstrap=bootstrap,
        seeds=out["seeds"],
        hits=hits,
        catalog=catalog,
    )
    return out
