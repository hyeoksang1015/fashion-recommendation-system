"""articles(상품 카탈로그) 정제 함수 모음.

역할: 비의류와 이미지 없는 상품을 빼고, type 이름과 garment group을 정리해
    모든 모델이 같은 상품 집합(valid_article_ids)을 쓰도록 한다.
동작: 각 함수는 DataFrame을 받아 새 DataFrame을 반환하며 입력을 수정하지 않는다.
    적용 순서는 src/pipeline/preprocess.py의 clean_articles에서 정한다
    (제외 그룹 -> 비의류 type -> 이미지 없음 -> type 통합 -> garment Unknown 매핑
    -> 희귀 type 병합 -> detail_desc 채우기 -> product_type_no 제거).
"""

import logging
import pandas as pd
from src.utils.image import find_existing_image_ids, get_image_paths

logger = logging.getLogger(__name__)

UNKNOWN = "Unknown"


def remove_excluded_groups(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """product_group_name이 제외 목록에 있는 상품을 제거한다.

    Args:
        df: articles DataFrame.
        groups: 제거할 product_group_name 목록.

    Returns:
        제외 그룹이 빠진 DataFrame.
    """
    return df[~df["product_group_name"].isin(groups)]


def remove_non_fashion_types(df: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    """product_type_name이 비의류 목록에 있는 상품을 제거한다.

    Args:
        df: articles DataFrame.
        types: 제거할 product_type_name 목록.

    Returns:
        비의류 type이 빠진 DataFrame.
    """
    return df[~df["product_type_name"].isin(types)]


def filter_articles_with_image(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """이미지가 없는 상품을 제거하고 image_path 컬럼을 추가한다.

    Args:
        df: articles DataFrame.
        image_dir: 이미지 최상위 디렉터리 (상대 경로).

    Returns:
        이미지가 있는 상품만 남고 image_path가 추가된 DataFrame.
    """
    existing = find_existing_image_ids(df["article_id"], image_dir)
    out = df[df["article_id"].isin(existing)].copy()
    out["image_path"] = get_image_paths(out["article_id"], image_dir)
    return out


def merge_product_types(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """type_merge_mapping으로 product_type_name을 통합한다.

    매핑의 key나 value가 데이터에 없으면 경고 로그를 남긴다.

    Args:
        df: articles DataFrame.
        mapping: {원래 type: 통합 type}.

    Returns:
        product_type_name이 통합된 DataFrame.
    """
    present = set(df["product_type_name"])
    missing_keys = set(mapping) - present
    missing_values = set(mapping.values()) - present
    if missing_keys:
        logger.warning(
            "type_merge_mapping key가 데이터에 없음: %s", sorted(missing_keys)
        )
    if missing_values:
        logger.warning(
            "type_merge_mapping value가 데이터에 없음: %s", sorted(missing_values)
        )
    out = df.copy()
    out["product_type_name"] = out["product_type_name"].replace(mapping)
    return out


def fill_unknown_garment_group(
    df: pd.DataFrame, mapping: dict[str, str]
) -> pd.DataFrame:
    """garment_group_name이 Unknown인 행에만 product_type_name 기준 매핑을 적용한다.

    희귀 type 병합보다 먼저 적용해야 한다 (병합 후에는 매핑 key가 사라진다).

    Args:
        df: articles DataFrame.
        mapping: {product_type_name: garment_group_name}.

    Returns:
        Unknown garment group이 채워진 DataFrame. 매핑이 없는 행은 Unknown 유지.

    Raises:
        ValueError: 매핑 값이 기존 garment_group_name 값에 없을 때.
    """
    invalid = set(mapping.values()) - set(df["garment_group_name"])
    if invalid:
        raise ValueError(f"garment_group_mapping 값이 기존 값에 없음: {invalid}")
    out = df.copy()
    mask = out["garment_group_name"] == UNKNOWN
    mapped = out.loc[mask, "product_type_name"].map(mapping)
    out.loc[mask, "garment_group_name"] = mapped.fillna(UNKNOWN)
    logger.info(
        "garment Unknown %d -> %d",
        mask.sum(),
        (out["garment_group_name"] == UNKNOWN).sum(),
    )
    return out


def find_rare_types(df: pd.DataFrame, min_count: int) -> pd.Series:
    """상품 수가 min_count 미만인 product_type_name과 상품 수를 반환한다.

    Args:
        df: articles DataFrame.
        min_count: 희귀 type 기준 상품 수.

    Returns:
        index=type 이름, value=상품 수인 Series.
    """
    counts = df["product_type_name"].value_counts()
    return counts[counts < min_count]


def merge_rare_types(df: pd.DataFrame, min_count: int, prefix: str) -> pd.DataFrame:
    """희귀 product_type_name을 f"{prefix}{product_group_name}"으로 바꾼다.

    상품은 제거하지 않는다. 거래가 아니라 상품 카탈로그 기준 개수라 누수가 없다.

    Args:
        df: articles DataFrame.
        min_count: 희귀 type 기준 상품 수.
        prefix: 병합 type 이름 접두사.

    Returns:
        희귀 type이 병합된 DataFrame.
    """
    rare = find_rare_types(df, min_count)
    logger.info("희귀 type %d개 병합: %s", len(rare), rare.to_dict())
    out = df.copy()
    is_rare = out["product_type_name"].isin(rare.index)
    out.loc[is_rare, "product_type_name"] = (
        prefix + out.loc[is_rare, "product_group_name"]
    )
    return out


def fill_detail_desc(df: pd.DataFrame) -> pd.DataFrame:
    """detail_desc 결측을 prod_name으로 채운다.

    Args:
        df: articles DataFrame.

    Returns:
        detail_desc 결측이 없는 DataFrame.
    """
    out = df.copy()
    out["detail_desc"] = out["detail_desc"].fillna(out["prod_name"])
    return out


def drop_product_type_no(df: pd.DataFrame) -> pd.DataFrame:
    """type 이름이 바뀌어 더 이상 맞지 않는 product_type_no를 제거한다.

    Args:
        df: articles DataFrame.

    Returns:
        product_type_no가 없는 DataFrame.
    """
    return df.drop(columns="product_type_no")
