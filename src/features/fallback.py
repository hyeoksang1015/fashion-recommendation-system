"""나이대별 fallback 추천 테이블.

역할: 개인화 추천을 줄 수 없는 유저(이력 없음 등)에게 줄 age_group별 top-k 목록을
    만든다.
동작: age_group마다 나이대별 인기 상위 k개를 쓰고, "Unknown" 그룹은 전체 인기 상위
    k개를 쓴다. 구매 이력이 없거나 k개보다 적은 그룹은 전체 인기로 중복 없이 채운다.
    루프는 age_group 수(7개)만큼만 돈다.
"""

import pandas as pd


def build_fallback_table(
    overall_popularity: pd.DataFrame,
    age_group_popularity: pd.DataFrame,
    k: int,
    unknown_label: str,
) -> dict[str, list[int]]:
    """age_group별 top-k article_id 목록을 만든다.

    Args:
        overall_popularity: compute_overall_popularity 결과 (순위순).
        age_group_popularity: compute_age_group_popularity 결과 (그룹 안 순위순).
        k: 목록 길이.
        unknown_label: 나이 결측 그룹 라벨. 이 그룹은 전체 인기를 쓴다.

    Returns:
        {age_group: [article_id(int), ...]}. 모든 age_group 라벨과 unknown_label을
        key로 가지며 각 목록은 길이 k (전체 인기 상품이 k개 미만이면 그만큼).

    Raises:
        ValueError: k가 1 미만일 때.
    """
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 함: {k}")
    overall = overall_popularity["article_id"].head(k).tolist()
    age_group = age_group_popularity["age_group"]
    labels = (
        list(age_group.cat.categories)
        if isinstance(age_group.dtype, pd.CategoricalDtype)
        else list(age_group.unique())
    )
    top = (
        age_group_popularity.groupby("age_group", observed=True, sort=False)
        .head(k)
        .groupby("age_group", observed=True, sort=False)["article_id"]
        .agg(list)
    )

    table = {}
    for label in labels:
        if label == unknown_label:
            continue
        items = top.get(label, [])
        # 이력이 부족한 그룹은 전체 인기로 채운다
        seen = set(items)
        items = items + [a for a in overall if a not in seen]
        table[label] = [int(a) for a in items[:k]]
    table[unknown_label] = overall
    return table
