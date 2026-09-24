"""YAML config 로드.

역할: configs/preprocess.yaml 같은 설정 파일을 dict로 읽어 파이프라인에 넘긴다.
동작: yaml.safe_load로 파일 전체를 읽는다. 경로, 임계값, 매핑은 모두 이 dict에서
    꺼내 쓴다.
"""

import yaml


def load_config(path: str) -> dict:
    """YAML config 파일을 dict로 읽는다.

    Args:
        path: config 파일 경로.

    Returns:
        config dict.

    Raises:
        FileNotFoundError: 파일이 없을 때.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
