"""ALS 학습/진단/실험 파이프라인.

역할: ALS 학습과 추천, 시작값 진단(재구매 hit 분해, 커버 유저 비교), 하이퍼파라미터
    조합 실험의 실행 순서를 정한다.
동작:
    - load_eval_context: 전처리 데이터, 정답셋, age_fallback 추천을 한 번 만든다.
    - fit_model / fit_als / fit_bpr: 행렬 생성 + 학습. 행렬과 모델은 추천 단계 설정
      (filter_already_purchased, k)과 무관하므로, 실험은 모델 설정이 같은 조합끼리
      묶어 한 번만 학습하고 추천만 조합별로 다시 한다.
    - ALS에 없는 유저는 age_fallback 목록으로 채운다.
"""

import itertools
import logging
import pandas as pd
import scipy.sparse as sp
import time
from collections.abc import Callable
from src.data.split import split_by_week
from src.evaluation.diagnostics import (
    decompose_hits,
    n_distinct_items,
    purchased_pairs,
    repurchase_rate,
)
from src.evaluation.ground_truth import build_ground_truth
from src.evaluation.metrics import evaluate_users
from src.models.als import build_interaction_matrix, recommend_als, train_als
from src.models.bpr import build_binary_matrix, train_bpr
from src.pipeline.baseline import (
    build_eval_groups,
    build_recommendations,
    evaluate_baseline,
    load_inputs,
    save_results,
    split_users,
)
from src.utils.config import load_config

logger = logging.getLogger(__name__)

BASELINE = "age_fallback"
MODEL = "als"
COVERED = "covered"
UNCOVERED = "uncovered"
# 추천 단계에만 쓰이는 키. 나머지 키가 같으면 행렬과 모델을 재사용한다.
RECOMMEND_KEYS = ("filter_already_purchased", "k")


def load_eval_context(config: dict) -> dict:
    """평가에 공통으로 쓰는 데이터와 age_fallback 추천을 만든다.

    Args:
        config: als config (processed_dir, target_split, features_config,
            evaluation_config).

    Returns:
        train, target, customers, articles, ground_truth, users, fallback,
        features_cfg, eval_cfg, split 키를 가진 dict.
    """
    split = config["target_split"]
    train, target, customers, articles = load_inputs(config["processed_dir"], split)
    return build_eval_context(config, train, target, customers, articles, split)


def build_eval_context(
    config: dict,
    train: pd.DataFrame,
    target: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    split: str,
) -> dict:
    """주어진 train / target으로 정답셋과 age_fallback 추천을 만든다.

    Args:
        config: als config (features_config, evaluation_config).
        train: train 거래 (customer_id, article_id, week_idx).
        target: 평가 거래 (customer_id, article_id).
        customers: 정제된 customers (customer_id, age_group).
        articles: 정제된 articles (article_id, product_group_name).
        split: 평가 분할 이름 (로그와 결과 표시용).

    Returns:
        load_eval_context와 같은 키의 dict.
    """
    features_cfg = load_config(config["features_config"])
    eval_cfg = load_config(config["evaluation_config"])
    ground_truth = build_ground_truth(target)
    users = list(ground_truth)
    logger.info("%s 정답셋: 유저 %d명", split, len(users))
    fallback = build_recommendations(users, train, customers, features_cfg)[BASELINE]
    return {
        "train": train,
        "target": target,
        "customers": customers,
        "articles": articles,
        "ground_truth": ground_truth,
        "users": users,
        "fallback": fallback,
        "features_cfg": features_cfg,
        "eval_cfg": eval_cfg,
        "split": split,
    }


def fit_model(
    train: pd.DataFrame,
    config: dict,
    build_matrix: Callable[[pd.DataFrame, dict], tuple],
    train_model: Callable[[sp.csr_matrix, dict], object],
) -> dict:
    """행렬을 만들고 implicit 행렬 분해 모델을 학습한다.

    Args:
        train: train 거래.
        config: 모델 config.
        build_matrix: (train, config) -> (matrix, user_ids, item_ids).
        train_model: (matrix, config) -> 학습된 모델.

    Returns:
        matrix, user_ids, item_ids, model, info 키를 가진 dict. info는 행렬 생성과
        학습 시간(초), 유저 수, 상품 수, 비제로 수.
    """
    start = time.perf_counter()
    matrix, user_ids, item_ids = build_matrix(train, config)
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model = train_model(matrix, config)
    train_seconds = time.perf_counter() - start
    return {
        "matrix": matrix,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "model": model,
        "info": {
            "build_matrix_seconds": round(build_seconds, 1),
            "train_seconds": round(train_seconds, 1),
            "n_users": matrix.shape[0],
            "n_items": matrix.shape[1],
            "nnz": int(matrix.nnz),
        },
    }


def fit_als(train: pd.DataFrame, config: dict) -> dict:
    """confidence 행렬을 만들고 ALS를 학습한다.

    Args:
        train: train 거래.
        config: als config (alpha, train_weeks, factors, regularization,
            iterations, seed).

    Returns:
        fit_model 결과.
    """
    return fit_model(
        train,
        config,
        lambda t, c: build_interaction_matrix(t, c["alpha"], c["train_weeks"]),
        train_als,
    )


def fit_bpr(train: pd.DataFrame, config: dict) -> dict:
    """구매 여부(0/1) 행렬을 만들고 BPR을 학습한다.

    Args:
        train: train 거래.
        config: bpr config (train_weeks, factors, learning_rate, regularization,
            iterations, seed, num_threads).

    Returns:
        fit_model 결과.
    """
    return fit_model(
        train,
        config,
        lambda t, c: build_binary_matrix(t, c["train_weeks"]),
        train_bpr,
    )


def apply_overrides(override_config: dict) -> dict:
    """base_config를 읽어 overrides를 덮어쓴 config를 만든다.

    Args:
        override_config: base_config(경로)와 overrides(dict)를 가진 config
            (예: configs/als_best.yaml).

    Returns:
        완성된 config.
    """
    return {
        **load_config(override_config["base_config"]),
        **override_config["overrides"],
    }


def recommend_with_fallback(
    fitted: dict, users: list[str], fallback: dict[str, list[int]], config: dict
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """ALS 추천을 만들고 ALS에 없는 유저는 fallback으로 채운다.

    Args:
        fitted: fit_model 결과 (ALS, BPR 등 implicit 행렬 분해 모델).
        users: 추천 대상 customer_id 목록.
        fallback: {customer_id: age_fallback 추천}.
        config: als config (k, filter_already_purchased).

    Returns:
        (ALS만의 추천, fallback으로 채운 전체 추천).
    """
    als_recs = recommend_als(
        fitted["model"],
        fitted["matrix"],
        fitted["user_ids"],
        fitted["item_ids"],
        users,
        config["k"],
        config["filter_already_purchased"],
    )
    return als_recs, {**fallback, **als_recs}


def _all_metrics(
    recs: dict[str, list[int]], ground_truth: dict[str, set[int]], eval_cfg: dict
) -> dict:
    """전체 지표에 평가 유저 수를 붙인다.

    Args:
        recs: {customer_id: 추천 리스트}.
        ground_truth: 평가할 정답셋.
        eval_cfg: evaluation config (k, metrics).

    Returns:
        {지표: 점수, "n_users": 유저 수}.
    """
    scores = evaluate_users(recs, ground_truth, eval_cfg["k"], eval_cfg["metrics"])
    return {**scores, "n_users": len(ground_truth)}


def hit_breakdown(
    ctx: dict,
    als_recs: dict[str, list[int]],
    recs: dict[str, list[int]],
    purchased: pd.DataFrame,
    k: int,
) -> dict[str, dict[str, int]]:
    """age_fallback, ALS(fallback 포함), ALS만의 hit를 재구매/신규 구매로 나눈다.

    Args:
        ctx: load_eval_context 결과 (ground_truth, fallback).
        als_recs: ALS 커버 유저의 ALS 추천.
        recs: fallback으로 채운 전체 추천.
        purchased: purchased_pairs 결과.
        k: 자를 순위.

    Returns:
        {"age_fallback" | "als_filled" | "als_only": decompose_hits 결과}.
        als_filled - als_only는 미커버 유저가 받은 fallback의 hit다.
    """
    models = {
        BASELINE: ctx["fallback"],
        f"{MODEL}_filled": recs,
        f"{MODEL}_only": als_recs,
    }
    return {
        name: decompose_hits(r, ctx["ground_truth"], purchased, k)
        for name, r in models.items()
    }


def split_by_coverage(
    ground_truth: dict[str, set[int]], als_recs: dict[str, list[int]]
) -> dict[str, dict[str, set[int]]]:
    """정답셋 유저를 ALS 커버(covered) / 미커버(uncovered)로 나눈다.

    Args:
        ground_truth: 전체 정답셋.
        als_recs: ALS 추천 (key가 커버 유저).

    Returns:
        {"covered" | "uncovered": 그 유저만 남은 ground_truth}.
    """
    return split_users(ground_truth, dict.fromkeys(als_recs, COVERED), UNCOVERED)


def evaluate_models(
    ctx: dict,
    models: dict[str, dict[str, list[int]]],
    item_metrics: list[str],
    extra_user_groups: dict | None = None,
) -> dict:
    """baseline과 같은 그룹 구조로 여러 추천을 평가한다.

    Args:
        ctx: load_eval_context 결과.
        models: {모델 이름: {customer_id: 추천 리스트}}.
        item_metrics: 상품 그룹 지표.
        extra_user_groups: 기본 유저 그룹(활동, 나이대)에 더할
            {그룹 종류: {라벨: ground_truth}}. None이면 더하지 않는다.

    Returns:
        {모델 이름: evaluate_baseline 결과}.
    """
    user_groups, item_groups = build_eval_groups(
        ctx["ground_truth"],
        ctx["train"],
        ctx["target"],
        ctx["customers"],
        ctx["articles"],
        ctx["features_cfg"],
    )
    user_groups.update(extra_user_groups or {})
    eval_cfg = ctx["eval_cfg"]
    return {
        name: evaluate_baseline(
            r,
            ctx["ground_truth"],
            user_groups,
            item_groups,
            eval_cfg["k"],
            eval_cfg["metrics"],
            item_metrics,
        )
        for name, r in models.items()
    }


def format_comparison(results: dict, metrics: list[str]) -> str:
    """전체 지표 표와 그룹별 recall 표를 문자열로 만든다.

    Args:
        results: {모델 이름: evaluate_baseline 결과}.
        metrics: 전체 표에 넣을 지표 이름.

    Returns:
        로그로 낼 표 문자열.
    """
    names = list(results)
    header = f"{'':<34}" + "".join(f"{n:>14}" for n in names)
    lines = ["[전체]", header]
    for m in metrics:
        row = "".join(f"{results[n]['all'][m]:>14.4f}" for n in names)
        lines.append(f"{m + '@k':<34}{row}")

    lines += ["", "[그룹별 recall@k]", header + f"{'n_users':>10}"]
    groups = {n: dict(_flatten_groups(results[n])) for n in names}
    for group, score in groups[names[0]].items():
        row = "".join(f"{groups[n][group].get('recall', 0.0):>14.4f}" for n in names)
        lines.append(f"{group:<34}{row}{score['n_users']:>10}")
    return "\n".join(lines)


def format_hits(hits: dict[str, dict[str, int]]) -> str:
    """hit_breakdown 결과를 표 문자열로 만든다.

    Args:
        hits: hit_breakdown 결과.

    Returns:
        모델을 열, hit 종류를 행으로 둔 표 문자열.
    """
    names = list(hits)
    lines = [f"{'':<22}" + "".join(f"{n:>14}" for n in names)]
    for key in hits[names[0]]:
        lines.append(f"{key:<22}" + "".join(f"{hits[n][key]:>14}" for n in names))
    return "\n".join(lines)


def _flatten_groups(result: dict) -> list[tuple[str, dict]]:
    """evaluate_baseline 결과의 그룹 점수를 (이름, 점수) 목록으로 편다.

    Args:
        result: evaluate_baseline 결과.

    Returns:
        유저 그룹은 "by_x/라벨", 상품 그룹은 그룹 이름으로 된 (이름, 점수) 목록.
    """
    rows = []
    for key, value in result.items():
        if key == "all":
            continue
        if "n_users" in value:  # 상품 그룹: 점수 dict 하나
            rows.append((key, value))
        else:  # 유저 그룹: {라벨: 점수}
            rows += [(f"{key}/{label}", score) for label, score in value.items()]
    return rows


def run_diagnosis(config: dict) -> dict:
    """ALS 시작값의 hit 구성과 커버 유저 성능을 age_fallback과 비교한다.

    Args:
        config: als config.

    Returns:
        {"hits": hit_breakdown 결과,
        "covered_users": {모델: 커버 유저만의 지표}, "coverage": 커버율}.
    """
    ctx = load_eval_context(config)
    k = ctx["eval_cfg"]["k"]
    fitted = fit_als(ctx["train"], config)
    als_recs, recs = recommend_with_fallback(
        fitted, ctx["users"], ctx["fallback"], config
    )
    purchased = purchased_pairs(ctx["train"], ctx["users"])
    models = {BASELINE: ctx["fallback"], MODEL: recs}
    covered = {u: ctx["ground_truth"][u] for u in als_recs}
    return {
        "hits": hit_breakdown(ctx, als_recs, recs, purchased, k),
        "covered_users": {
            name: _all_metrics(r, covered, ctx["eval_cfg"])
            for name, r in models.items()
        },
        "coverage": len(als_recs) / len(ctx["users"]),
    }


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """{키: 값 목록}의 모든 조합을 만든다.

    Args:
        grid: {config 키: 후보 값 목록}.

    Returns:
        조합 dict 목록 (grid 키 순서의 곱집합 순서).
    """
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def group_by_model(configs: list[dict]) -> dict[tuple, list[dict]]:
    """추천 단계 키를 뺀 설정이 같은 config끼리 묶는다.

    Args:
        configs: 완성된 als config 목록.

    Returns:
        {모델 설정 key: 그 설정을 쓰는 config 목록}. 등장 순서를 지킨다.
    """
    groups: dict[tuple, list[dict]] = {}
    for cfg in configs:
        key = tuple(
            sorted((k, repr(v)) for k, v in cfg.items() if k not in RECOMMEND_KEYS)
        )
        groups.setdefault(key, []).append(cfg)
    return groups


def run_experiments(exp_config: dict) -> list[dict]:
    """als config 기본값 위에 조합을 덮어써 실험하고 json으로 저장한다.

    Args:
        exp_config: 실험 config (base_config, output_path, grid).

    Returns:
        조합별 {params, all, coverage, n_distinct_items, repurchase_rate,
        fit_info} 목록.
    """
    base = load_config(exp_config["base_config"])
    ctx = load_eval_context(base)
    k = ctx["eval_cfg"]["k"]
    purchased = purchased_pairs(ctx["train"], ctx["users"])
    combos = expand_grid(exp_config["grid"])

    records = []
    for configs in group_by_model([{**base, **c} for c in combos]).values():
        fitted = fit_als(ctx["train"], configs[0])
        logger.info("학습 완료 %s", fitted["info"])
        for cfg in configs:
            als_recs, recs = recommend_with_fallback(
                fitted, ctx["users"], ctx["fallback"], cfg
            )
            record = {
                "params": {key: cfg[key] for key in exp_config["grid"]},
                "all": _all_metrics(recs, ctx["ground_truth"], ctx["eval_cfg"]),
                "coverage": len(als_recs) / len(ctx["users"]),
                "n_distinct_items": n_distinct_items(als_recs),
                "repurchase_rate": repurchase_rate(als_recs, purchased, k),
                "fit_info": fitted["info"],
            }
            logger.info("%s -> %s", record["params"], record["all"])
            records.append(record)
        del fitted

    save_results(
        exp_config["output_path"],
        ctx["split"],
        k,
        {BASELINE: _all_metrics(ctx["fallback"], ctx["ground_truth"], ctx["eval_cfg"])},
        experiments=records,
    )
    return records


def shift_eval_split(
    train: pd.DataFrame, test_weeks: int, valid_weeks: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train 안에서 평가 주를 과거로 옮긴 (train, target)을 만든다.

    전처리와 같은 split_by_week 규칙을 train에 다시 적용한다. week_idx < test_weeks
    구간은 버리고, 그다음 valid_weeks주를 target, 나머지를 train으로 쓴다.

    Args:
        train: 원래 train 거래 (week_idx 포함).
        test_weeks: 버릴 최근 주 수 (week_idx 기준 경계).
        valid_weeks: 평가에 쓸 주 수.

    Returns:
        (새 train, 새 target).

    Raises:
        ValueError: 새 target이나 새 train이 비었을 때.
    """
    new_train, new_target, _ = split_by_week(train, test_weeks, valid_weeks)
    if new_target.empty or new_train.empty:
        raise ValueError(
            f"재확인 분할이 비었음: test_weeks={test_weeks}, valid_weeks={valid_weeks}"
        )
    return new_train, new_target


def run_best_validation(val_config: dict) -> dict:
    """최고 조합을 상세 평가하고, 평가 주를 옮긴 분할로 재확인한다.

    Args:
        val_config: 검증 config (base_config, overrides, output_path, recheck).

    Returns:
        {"results": 모델별 그룹 평가(ALS 커버/미커버 그룹 포함),
        "hits": hit_breakdown 결과, "als_info": 학습 정보와 커버율,
        "recheck": 재확인 평가의 all 지표와 커버율}.
    """
    config = apply_overrides(val_config)
    ctx = load_eval_context(config)
    k = ctx["eval_cfg"]["k"]
    fitted = fit_als(ctx["train"], config)
    als_recs, recs = recommend_with_fallback(
        fitted, ctx["users"], ctx["fallback"], config
    )
    fit_info = fitted["info"]
    del fitted
    results = evaluate_models(
        ctx,
        {BASELINE: ctx["fallback"], MODEL: recs},
        config["item_group_metrics"],
        {"by_als_coverage": split_by_coverage(ctx["ground_truth"], als_recs)},
    )
    purchased = purchased_pairs(ctx["train"], ctx["users"])
    hits = hit_breakdown(ctx, als_recs, recs, purchased, k)
    als_info = {
        "params": val_config["overrides"],
        "coverage": len(als_recs) / len(ctx["users"]),
        "fit_info": fit_info,
    }

    recheck_cfg = val_config["recheck"]
    train, target = shift_eval_split(
        ctx["train"], recheck_cfg["test_weeks"], recheck_cfg["valid_weeks"]
    )
    weeks = sorted(target["week_idx"].unique().tolist())
    rctx = build_eval_context(
        config, train, target, ctx["customers"], ctx["articles"], f"week_idx {weeks}"
    )
    rfitted = fit_als(train, config)
    r_als, r_recs = recommend_with_fallback(
        rfitted, rctx["users"], rctx["fallback"], config
    )
    recheck = {
        **recheck_cfg,
        "target_week_idx": weeks,
        "train_week_idx_min": int(train["week_idx"].min()),
        "coverage": len(r_als) / len(rctx["users"]),
        "fit_info": rfitted["info"],
        "results": {
            name: _all_metrics(r, rctx["ground_truth"], rctx["eval_cfg"])
            for name, r in {BASELINE: rctx["fallback"], MODEL: r_recs}.items()
        },
    }

    save_results(
        val_config["output_path"],
        ctx["split"],
        k,
        results,
        hits=hits,
        als_info=als_info,
        recheck=recheck,
    )
    return {"results": results, "hits": hits, "als_info": als_info, "recheck": recheck}
