"""이미지 임베딩 기반 유저 프로필 추천."""

import numpy as np
import os
import pandas as pd
from scipy import sparse


def align_embeddings(
    item_ids: np.ndarray,
    all_ids: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """item_ids 순서에 맞춰 임베딩 행을 재정렬한다.

    Description:
        pd.Index.get_indexer로 값을 위치로 한 번에 바꿔 O(N)에 처리한다.

    Args:
        item_ids: 목표 순서의 article_id 배열 (예: 행렬 열 순서).
        all_ids: embeddings 각 행에 대응하는 article_id 배열.
        embeddings: (len(all_ids), dim) 임베딩.

    Returns:
        (len(item_ids), dim) 임베딩.

    Raises:
        ValueError: item_ids 중 all_ids에 없는 상품이 있을 때.
    """
    rows = pd.Index(all_ids).get_indexer(item_ids)
    missing = int((rows < 0).sum())
    if missing:
        raise ValueError(f"임베딩이 없는 상품 {missing}개")
    return embeddings[rows]


def build_user_profiles(
    interactions: sparse.csr_matrix,
    item_embeddings: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """구매 이력을 이진화해 유저 프로필 벡터를 만든다.

    Description:
        유저가 산 상품 임베딩의 합을 L2 정규화한다. 이진화로 구매
        횟수의 영향을 제거해 특정 상품 쪽으로 프로필이 쏠리지 않게 한다.

    Args:
        interactions: (유저, 상품) 희소 행렬. 0이 아닌 값은 모두 구매.
        item_embeddings: (상품, dim) L2 정규화 임베딩. 열 순서가 일치해야 함.
        eps: 0으로 나누는 것을 막는 하한.

    Returns:
        (유저, dim) float32 프로필. 이력이 없는 유저는 영벡터.

    Raises:
        ValueError: 행렬 열 수와 임베딩 행 수가 다를 때.
    """
    if interactions.shape[1] != item_embeddings.shape[0]:
        raise ValueError("행렬 열 수와 임베딩 행 수가 다릅니다")
    binary = sparse.csr_matrix(interactions, dtype=np.float32, copy=True)
    binary.data[:] = 1.0
    profiles = binary @ item_embeddings
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    return (profiles / np.maximum(norms, eps)).astype(np.float32)


def recommend_by_image(
    profiles: np.ndarray,
    item_embeddings: np.ndarray,
    k: int = 12,
    chunk_size: int = 512,
) -> np.ndarray:
    """프로필과 유사도가 높은 상위 k개 상품의 행 인덱스를 반환한다.

    Description:
        유저를 청크로 나눠 내적을 계산하고 argpartition으로 O(N)에
        상위 k개를 모은 뒤, 그 k개만 정렬한다.

    Args:
        profiles: (유저, dim) 프로필.
        item_embeddings: (후보 상품, dim) 전체 임베딩.
        k: 추천 개수.
        chunk_size: 한 번에 계산할 유저 수. 메모리 상한을 정한다.

    Returns:
        (유저, k) int64 배열. 유사도 내림차순, 값은 임베딩 행 인덱스.

    Raises:
        ValueError: k가 후보 수 이상일 때.
    """
    if k >= item_embeddings.shape[0]:
        raise ValueError("k는 후보 수보다 작아야 합니다")
    result = np.empty((profiles.shape[0], k), dtype=np.int64)
    for start in range(0, profiles.shape[0], chunk_size):
        end = start + chunk_size
        sims = profiles[start:end] @ item_embeddings.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_sims, axis=1)
        result[start:end] = np.take_along_axis(top, order, axis=1)
    return result


def load_embeddings(
    embedding_dir: str, sample: int = 1000, atol: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """저장된 article_id와 임베딩을 읽고 형태를 검증한다.

    Description:
        두 파일은 같은 행 순서로 저장돼 있어야 한다. 앞쪽 일부 행의 노름이
        1인지 확인해 정규화 누락(내적이 코사인이 아니게 되는 문제)을 잡는다.

    Args:
        embedding_dir: article_ids.npy, embeddings.npy가 있는 폴더.
        sample: 정규화를 확인할 앞쪽 행 수.
        atol: 노름이 1에서 벗어나도 허용하는 오차.

    Returns:
        (article_ids int64 (N,), embeddings (N, dim)).

    Raises:
        FileNotFoundError: 파일이 없을 때.
        ValueError: 행 수가 다르거나 L2 정규화가 되어 있지 않을 때.
    """
    ids_path = os.path.join(embedding_dir, "article_ids.npy")
    emb_path = os.path.join(embedding_dir, "embeddings.npy")
    article_ids = np.load(ids_path).astype(np.int64)
    embeddings = np.load(emb_path)
    if embeddings.shape[0] != article_ids.shape[0]:
        raise ValueError("article_ids와 embeddings의 행 수가 다릅니다")
    if not np.allclose(np.linalg.norm(embeddings[:sample], axis=1), 1.0, atol=atol):
        raise ValueError("임베딩이 L2 정규화되어 있지 않습니다")
    return article_ids, embeddings


def recommend_for_users(
    matrix: sparse.csr_matrix,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    article_ids: np.ndarray,
    embeddings: np.ndarray,
    users: list[str],
    k: int = 12,
    chunk_size: int = 512,
) -> dict[str, list[int]]:
    """학습 행렬에 이력이 있는 유저에게 이미지 프로필 추천을 만든다.

    Description:
        행렬 행으로 프로필을 만들고, 후보는 행렬 열(ALS 후보)이 아니라
        임베딩 전체로 둔다. 행렬에 없는 유저는 결과에서 빠진다.

    Args:
        matrix: (유저, 상품) 학습 행렬.
        user_ids: matrix 행에 대응하는 customer_id.
        item_ids: matrix 열에 대응하는 article_id.
        article_ids: embeddings 각 행에 대응하는 article_id.
        embeddings: (전체 상품, dim) L2 정규화 임베딩.
        users: 추천 대상 customer_id 목록.
        k: 추천 개수.
        chunk_size: recommend_by_image의 청크 크기.

    Returns:
        {customer_id: [article_id(int), ...]}.

    Raises:
        ValueError: 행렬 상품이 임베딩에 없거나 k가 후보 수 이상일 때.
    """
    rows = pd.Index(user_ids).get_indexer(users)
    found = rows >= 0
    item_emb = align_embeddings(item_ids, article_ids, embeddings)
    profiles = build_user_profiles(matrix[rows[found]], item_emb)
    top = recommend_by_image(profiles, embeddings, k, chunk_size)
    kept = np.asarray(users, dtype=object)[found]
    return dict(zip(kept, article_ids[top].tolist()))
