"""H&M 상품 이미지 경로 계산과 존재 확인.

역할: article_id를 이미지 파일 경로로 바꾸고, 실제 파일이 있는 상품을 골라낸다.
동작:
    - 경로 규칙: {image_dir}/{10자리 id 앞 3자리}/{10자리 id}.jpg
      (정수 article_id는 앞자리 0이 빠지므로 10자리로 zero-padding한다)
    - 존재 확인은 필요한 하위 폴더만 os.scandir로 한 번씩 읽어 파일명 set을 만들고,
      벡터 연산(isin)으로 조회한다. 상품마다 os.path.exists를 호출하지 않는다.
"""

import os
import pandas as pd


def get_image_path(article_id: int, image_dir: str) -> str:
    """article_id의 이미지 경로를 만든다.

    Args:
        article_id: 상품 id (정수, 앞자리 0 없음).
        image_dir: 이미지 최상위 디렉터리.

    Returns:
        이미지 경로 문자열. 예: 108775015 -> {image_dir}/010/0108775015.jpg
    """
    padded = f"{article_id:010d}"
    return f"{image_dir}/{padded[:3]}/{padded}.jpg"


def get_image_paths(article_ids: pd.Series, image_dir: str) -> pd.Series:
    """article_id Series 전체의 이미지 경로를 벡터 연산으로 만든다.

    Args:
        article_ids: 상품 id Series.
        image_dir: 이미지 최상위 디렉터리.

    Returns:
        이미지 경로 문자열 Series (입력과 같은 index).
    """
    padded = article_ids.astype(str).str.zfill(10)
    return image_dir + "/" + padded.str[:3] + "/" + padded + ".jpg"


def find_existing_image_ids(article_ids: pd.Series, image_dir: str) -> set[int]:
    """실제 이미지 파일이 있는 article_id 집합을 반환한다.

    필요한 하위 폴더만 os.scandir로 한 번씩 읽어 파일명 set을 만든 뒤 조회한다.

    Args:
        article_ids: 확인할 상품 id Series.
        image_dir: 이미지 최상위 디렉터리.

    Returns:
        이미지가 존재하는 article_id 집합.
    """
    padded = article_ids.astype(str).str.zfill(10)
    file_names = set()
    for folder in padded.str[:3].unique():
        folder_path = os.path.join(image_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        with os.scandir(folder_path) as entries:
            file_names.update(entry.name for entry in entries)
    exists = (padded + ".jpg").isin(file_names)
    return set(article_ids[exists.to_numpy()].tolist())
