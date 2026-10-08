"""src.features.image_profile 단위 테스트."""

import numpy as np
import pytest
from scipy import sparse
from src.features.image_profile import (
    align_embeddings,
    build_user_profiles,
    load_embeddings,
    recommend_by_image,
    recommend_for_users,
)


def _unit_embeddings(n_items, dim, seed=0):
    """L2 정규화된 임의 임베딩을 만든다."""
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n_items, dim)).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def test_align_embeddings_reorders_rows():
    all_ids = np.array([10, 20, 30])
    emb = np.eye(3, dtype=np.float32)
    out = align_embeddings(np.array([30, 10]), all_ids, emb)
    np.testing.assert_array_equal(out, emb[[2, 0]])


def test_align_embeddings_raises_on_missing_id():
    with pytest.raises(ValueError):
        align_embeddings(
            np.array([10, 99]), np.array([10, 20]), np.eye(2, dtype=np.float32)
        )


def test_profiles_ignore_purchase_count():
    emb = _unit_embeddings(4, 8)
    once = sparse.csr_matrix(np.array([[1, 1, 0, 0]], dtype=np.float32))
    five = sparse.csr_matrix(np.array([[5, 1, 0, 0]], dtype=np.float32))
    np.testing.assert_allclose(
        build_user_profiles(once, emb), build_user_profiles(five, emb), atol=1e-6
    )


def test_profiles_zero_for_empty_user_and_unit_norm_otherwise():
    emb = _unit_embeddings(4, 8)
    mat = sparse.csr_matrix(np.array([[0, 0, 0, 0], [1, 0, 1, 0]], dtype=np.float32))
    prof = build_user_profiles(mat, emb)
    assert np.all(prof[0] == 0)
    assert np.linalg.norm(prof[1]) == pytest.approx(1.0, abs=1e-5)


def test_profiles_raises_on_shape_mismatch():
    emb = _unit_embeddings(4, 8)
    with pytest.raises(ValueError):
        build_user_profiles(sparse.csr_matrix(np.ones((2, 5))), emb)


def test_profiles_do_not_modify_input():
    emb = _unit_embeddings(3, 4)
    mat = sparse.csr_matrix(np.array([[3, 0, 2]], dtype=np.float32))
    build_user_profiles(mat, emb)
    np.testing.assert_array_equal(mat.data, [3.0, 2.0])


def test_recommend_matches_full_argsort():
    emb = _unit_embeddings(200, 16)
    prof = _unit_embeddings(30, 16, seed=1)
    expected = np.argsort(-(prof @ emb.T), axis=1)[:, :12]
    np.testing.assert_array_equal(recommend_by_image(prof, emb, k=12), expected)


def test_recommend_independent_of_chunk_size():
    emb = _unit_embeddings(200, 16)
    prof = _unit_embeddings(30, 16, seed=1)
    np.testing.assert_array_equal(
        recommend_by_image(prof, emb, k=5, chunk_size=7),
        recommend_by_image(prof, emb, k=5, chunk_size=512),
    )


def test_recommend_raises_when_k_not_less_than_candidates():
    emb = _unit_embeddings(5, 4)
    with pytest.raises(ValueError):
        recommend_by_image(emb, emb, k=5)


def test_load_embeddings_roundtrip_and_validation(tmp_path):
    emb = _unit_embeddings(6, 4)
    np.save(tmp_path / "embeddings.npy", emb)
    np.save(tmp_path / "article_ids.npy", np.arange(6, dtype=np.int32))
    ids, loaded = load_embeddings(str(tmp_path))
    assert ids.dtype == np.int64
    np.testing.assert_array_equal(loaded, emb)

    np.save(tmp_path / "embeddings.npy", emb * 2)
    with pytest.raises(ValueError):
        load_embeddings(str(tmp_path))

    np.save(tmp_path / "embeddings.npy", emb[:5])
    with pytest.raises(ValueError):
        load_embeddings(str(tmp_path))


def test_recommend_for_users_uses_full_candidates_and_skips_unknown():
    article_ids = np.array([40, 30, 20, 10, 50])
    emb = _unit_embeddings(5, 8)
    item_ids = np.array([10, 20, 30, 40])
    matrix = sparse.csr_matrix(
        np.array([[1, 0, 0, 0], [0, 2, 0, 1], [0, 0, 1, 0]], dtype=np.float32)
    )
    recs = recommend_for_users(
        matrix,
        np.array(["a", "b", "c"], dtype=object),
        item_ids,
        article_ids,
        emb,
        ["b", "zzz", "a"],
        k=3,
    )
    assert list(recs) == ["b", "a"]
    valid = set(article_ids.tolist())
    assert all(len(r) == 3 and set(r) <= valid for r in recs.values())
    # 유저 a는 상품 10만 샀으므로 프로필이 10과 같아 1위는 10 자신이다
    assert recs["a"][0] == 10
