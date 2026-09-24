"""이미지 경로 테스트.

10자리 zero-padding 경로 규칙, 벡터 버전과 단일 버전의 결과 일치,
os.scandir 기반 존재 확인을 임시 폴더(tmp_path)로 확인한다.
"""

import pandas as pd
from src.utils.image import find_existing_image_ids, get_image_path, get_image_paths


def test_get_image_path_zero_padding():
    assert get_image_path(108775015, "images") == "images/010/0108775015.jpg"


def test_get_image_paths_matches_single():
    ids = pd.Series([108775015, 956217002])
    assert get_image_paths(ids, "images").tolist() == [
        get_image_path(108775015, "images"),
        get_image_path(956217002, "images"),
    ]


def test_find_existing_image_ids(tmp_path):
    (tmp_path / "010").mkdir()
    (tmp_path / "010" / "0108775015.jpg").touch()
    ids = pd.Series([108775015, 108775044, 956217002], dtype="int32")
    assert find_existing_image_ids(ids, str(tmp_path)) == {108775015}
