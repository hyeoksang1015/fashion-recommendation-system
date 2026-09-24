"""raw CSV 로드.

역할: data/raw의 articles, customers, transactions CSV를 DataFrame으로 읽는다.
동작: read_csv에 dtype을 미리 지정해 읽는 시점부터 메모리를 줄인다
    (article_id int32, price float32, sales_channel_id int8,
    거래의 customer_id category). config의 transactions_nrows가 있으면
    거래는 앞에서부터 그만큼만 읽는다(개발용).
"""

import os
import pandas as pd

TRANSACTIONS_DTYPES = {
    "customer_id": "category",
    "article_id": "int32",
    "price": "float32",
    "sales_channel_id": "int8",
}


def load_articles(path: str) -> pd.DataFrame:
    """articles CSV를 읽는다.

    Args:
        path: articles.csv 경로.

    Returns:
        article_id가 int32인 articles DataFrame.

    Raises:
        FileNotFoundError: 파일이 없을 때.
    """
    return pd.read_csv(path, dtype={"article_id": "int32"})


def load_customers(path: str) -> pd.DataFrame:
    """customers CSV를 읽는다.

    Args:
        path: customers.csv 경로.

    Returns:
        customer_id가 string인 customers DataFrame.

    Raises:
        FileNotFoundError: 파일이 없을 때.
    """
    return pd.read_csv(path, dtype={"customer_id": "string"})


def load_transactions(path: str, nrows: int | None = None) -> pd.DataFrame:
    """transactions CSV를 읽는다.

    Args:
        path: transactions_train.csv 경로.
        nrows: 앞에서부터 읽을 행 수. None이면 전체.

    Returns:
        t_dat이 datetime이고 나머지 컬럼이 축소된 dtype인 DataFrame.

    Raises:
        FileNotFoundError: 파일이 없을 때.
    """
    return pd.read_csv(
        path, dtype=TRANSACTIONS_DTYPES, parse_dates=["t_dat"], nrows=nrows
    )


def load_raw(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """config의 경로로 articles, customers, transactions를 읽는다.

    Args:
        config: preprocess config dict.

    Returns:
        (articles, customers, transactions) 튜플.

    Raises:
        FileNotFoundError: 파일이 없을 때.
    """
    paths = config["paths"]
    raw_dir = paths["raw_dir"]
    articles = load_articles(os.path.join(raw_dir, paths["articles_file"]))
    customers = load_customers(os.path.join(raw_dir, paths["customers_file"]))
    transactions = load_transactions(
        os.path.join(raw_dir, paths["transactions_file"]),
        nrows=config["load"]["transactions_nrows"],
    )
    return articles, customers, transactions
