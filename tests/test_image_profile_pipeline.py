"""run_image_profile 연결 테스트.

가짜 전처리 parquet과 임베딩으로 ALS 학습, 이미지 추천, 평가까지 전체를 실행한다.
implicit이 없는 환경에서는 건너뛴다.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_USERS, N_ITEMS, N_NEW, N_COVERED = 30, 40, 5, 20
BASE_ID = 1000


def _write_data(tmp_path: Path) -> dict:
    """가짜 processed parquet, 임베딩, config yaml을 만들고 config를 돌려준다.

    u0~u19는 최근 4주(week 2~5)에만, 나머지는 week 8~9에만 구매한다. 정답셋에는
    train에 없는 신상품(N_NEW개)이 섞여 있다.
    """
    rng = np.random.default_rng(0)
    popularity = 1.0 / (np.arange(N_ITEMS) + 1)
    popularity /= popularity.sum()
    rows = []
    for u in range(N_USERS):
        for week in range(2, 6) if u < N_COVERED else range(8, 10):
            for item in rng.choice(N_ITEMS, size=3, replace=False, p=popularity):
                rows.append((f"u{u}", BASE_ID + item, week))
    train = pd.DataFrame(rows, columns=["customer_id", "article_id", "week_idx"])
    train = train.astype(
        {"customer_id": "category", "article_id": "int32", "week_idx": "int16"}
    )
    valid = pd.DataFrame(
        [
            (f"u{u}", BASE_ID + item)
            for u in range(N_USERS)
            for item in rng.choice(N_ITEMS + N_NEW, size=3, replace=False)
        ],
        columns=["customer_id", "article_id"],
    ).astype({"customer_id": "category", "article_id": "int32"})
    customers = pd.DataFrame(
        {
            "customer_id": pd.array([f"u{u}" for u in range(N_USERS)], dtype="string"),
            "age_group": rng.choice(["20s", "30s"], size=N_USERS),
        }
    )
    ids = BASE_ID + np.arange(N_ITEMS + N_NEW, dtype=np.int32)
    articles = pd.DataFrame({"article_id": ids, "product_group_name": "Upper body"})

    processed = tmp_path / "processed"
    processed.mkdir()
    train.to_parquet(processed / "transactions_train.parquet")
    valid.to_parquet(processed / "transactions_valid.parquet")
    customers.to_parquet(processed / "customers.parquet")
    articles.to_parquet(processed / "articles.parquet")

    shuffled = rng.permutation(ids)  # 임베딩 행 순서와 행렬 열 순서가 다르게
    emb = rng.normal(size=(len(shuffled), 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    np.save(emb_dir / "embeddings.npy", emb)
    np.save(emb_dir / "article_ids.npy", shuffled)

    als = yaml.safe_load((ROOT / "configs/als.yaml").read_text(encoding="utf-8"))
    als.update(
        processed_dir=str(processed),
        features_config=str(ROOT / "configs/features.yaml"),
        evaluation_config=str(ROOT / "configs/evaluation.yaml"),
        output_dir=str(tmp_path / "out"),
        factors=4,
        iterations=2,
    )
    (tmp_path / "als.yaml").write_text(yaml.safe_dump(als), encoding="utf-8")
    best = {
        "base_config": str(tmp_path / "als.yaml"),
        "overrides": {"train_weeks": 4, "filter_already_purchased": False},
    }
    (tmp_path / "als_best.yaml").write_text(yaml.safe_dump(best), encoding="utf-8")
    return {
        "als_config": str(tmp_path / "als_best.yaml"),
        "embedding_dir": str(emb_dir),
        "item_group_metrics": ["recall", "map"],
        "chunk_size": 7,
    }


def test_run_image_profile_end_to_end(tmp_path):
    pytest.importorskip("implicit")
    from src.pipeline.image_profile import run_image_profile

    out = run_image_profile(_write_data(tmp_path))

    models = {"als", "image_profile", "als_filter", "image_filter"}
    assert set(out["results"]) == models
    assert set(out["diagnostics"]) == models
    assert out["info"]["covered_users"] == N_COVERED
    for name in models:
        assert out["results"][name]["all"]["n_users"] == N_COVERED
    # 커버 유저는 모두 학습 4주 안에서만 샀으므로 필터 모델의 재구매 비율은 0이다
    assert out["diagnostics"]["image_filter"]["repurchase_rate"] == 0.0
    assert out["diagnostics"]["als_filter"]["repurchase_rate"] == 0.0
    # 프로필이 산 상품과 가장 비슷하므로 필터가 없으면 재구매가 많다
    assert out["diagnostics"]["image_profile"]["repurchase_rate"] > 0.0
