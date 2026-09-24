"""articles 정제 테스트.

비의류 제거, type 통합(없는 매핑 key 경고), garment 매핑이 Unknown 행에만
적용되는지, 희귀 type의 Other_{group} 병합, garment 매핑이 희귀 병합보다 먼저여야
하는 이유를 수십 행짜리 가짜 DataFrame으로 확인한다.
"""

import pandas as pd
import pytest
from src.data.articles import (
    fill_detail_desc,
    fill_unknown_garment_group,
    merge_product_types,
    merge_rare_types,
    remove_excluded_groups,
    remove_non_fashion_types,
)


def make_articles(rows):
    cols = [
        "product_type_name",
        "product_group_name",
        "garment_group_name",
        "prod_name",
        "detail_desc",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_remove_non_fashion():
    df = make_articles(
        [
            ["Umbrella", "Accessories", "Accessories", "a", "d"],
            ["Sofa", "Furniture", "Unknown", "b", "d"],
            ["Dress", "Garment Full body", "Dresses Ladies", "c", "d"],
        ]
    )
    out = remove_non_fashion_types(
        remove_excluded_groups(df, ["Furniture"]), ["Umbrella"]
    )
    assert out["product_type_name"].tolist() == ["Dress"]


def test_merge_product_types_warns_missing(caplog):
    df = make_articles(
        [
            ["Backpack", "Accessories", "Accessories", "a", "d"],
            ["Bag", "Accessories", "Accessories", "b", "d"],
        ]
    )
    out = merge_product_types(df, {"Backpack": "Bag", "Tyop": "Bag"})
    assert out["product_type_name"].tolist() == ["Bag", "Bag"]
    assert "Tyop" in caplog.text
    assert df["product_type_name"].tolist() == ["Backpack", "Bag"]  # 입력 불변


def test_garment_mapping_only_unknown_rows():
    df = make_articles(
        [
            ["T-shirt", "Garment Upper body", "Unknown", "a", "d"],
            ["T-shirt", "Garment Upper body", "Jersey Fancy", "b", "d"],
            ["Blouse", "Garment Upper body", "Unknown", "c", "d"],
            ["Blouse", "Garment Upper body", "Blouses", "c", "d"],
            ["Jersey", "Garment Upper body", "Jersey Basic", "c", "d"],
        ]
    )
    mapping = {"T-shirt": "Jersey Basic"}
    out = fill_unknown_garment_group(df, mapping)
    assert out["garment_group_name"].tolist() == [
        "Jersey Basic",
        "Jersey Fancy",
        "Unknown",  # 매핑 없음 -> 유지
        "Blouses",
        "Jersey Basic",
    ]


def test_garment_mapping_invalid_value_raises():
    df = make_articles([["T-shirt", "g", "Unknown", "a", "d"]])
    with pytest.raises(ValueError):
        fill_unknown_garment_group(df, {"T-shirt": "Nope"})


def test_rare_types_become_other_group():
    rows = [["Dress", "Garment Full body", "Dresses Ladies", "a", "d"]] * 3 + [
        ["Heels", "Shoes", "Shoes", "b", "d"]
    ]
    out = merge_rare_types(make_articles(rows), min_count=2, prefix="Other_")
    assert out["product_type_name"].tolist() == ["Dress"] * 3 + ["Other_Shoes"]
    assert len(out) == 4  # 상품은 제거하지 않음


def test_garment_mapping_before_rare_merge():
    # Bootie는 희귀 type이다. garment 매핑을 먼저 해야 Shoes로 채워진다.
    rows = [["Bootie", "Shoes", "Unknown", "a", "d"]] + [
        ["Sneakers", "Shoes", "Shoes", "b", "d"]
    ] * 3
    df = make_articles(rows)
    out = merge_rare_types(
        fill_unknown_garment_group(df, {"Bootie": "Shoes"}), 2, "Other_"
    )
    assert out.loc[0, "garment_group_name"] == "Shoes"
    assert out.loc[0, "product_type_name"] == "Other_Shoes"

    wrong = fill_unknown_garment_group(
        merge_rare_types(df, 2, "Other_"), {"Bootie": "Shoes"}
    )
    assert wrong.loc[0, "garment_group_name"] == "Unknown"


def test_fill_detail_desc():
    df = make_articles([["Dress", "g", "d", "name", None]])
    assert fill_detail_desc(df).loc[0, "detail_desc"] == "name"
