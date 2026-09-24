"""top-k 추천 평가 지표.

역할: 유저별 추천 리스트와 정답 집합으로 precision/recall/nDCG/MAP@k를 계산한다.
동작: 유저 루프 + set 멤버십 조회로 계산한다(전체 O(유저 수 x k)). 유저마다 정답
    크기가 달라 dense 행렬보다 set 조회가 효율적이다.
    - relevance는 binary (구매 1, 미구매 0)
    - 추천 리스트에 같은 상품이 반복되면 첫 번째만 hit로 센다
    - AP@k 분모는 min(정답 수, k) (H&M 대회 MAP@12 정의)
    - 평가 대상은 정답셋에 있는(구매가 있는) 유저뿐이다
    - article_id는 정수(parquet dtype 그대로). 추천과 정답의 타입이 다르면 에러
"""

import math
import numbers


def _hit_flags(recommended: list[int], relevant: set[int], k: int) -> list[bool]:
    """상위 k개 추천의 순위별 hit 여부. 중복 추천은 첫 번째만 hit.

    Args:
        recommended: 순위순 추천 리스트.
        relevant: 정답 집합.
        k: 자를 순위.

    Returns:
        길이 min(k, len(recommended))의 hit 여부 리스트.
    """
    seen = set()
    flags = []
    for item in recommended[:k]:
        flags.append(item in relevant and item not in seen)
        seen.add(item)
    return flags


def _check(relevant: set[int], k: int) -> None:
    """정답 집합과 k를 검사한다.

    Args:
        relevant: 정답 집합.
        k: 자를 순위.

    Raises:
        ValueError: 정답 집합이 비었거나 k가 1 미만일 때.
    """
    if not relevant:
        raise ValueError("relevant가 비어 있음: 구매가 없는 유저는 평가 대상이 아님")
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 함: {k}")


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Precision@k = 상위 k개 중 hit 수 / k.

    Args:
        recommended: 순위순 추천 리스트.
        relevant: 정답 집합.
        k: 자를 순위.

    Returns:
        0~1 사이 precision.

    Raises:
        ValueError: 정답 집합이 비었거나 k가 1 미만일 때.
    """
    _check(relevant, k)
    return sum(_hit_flags(recommended, relevant, k)) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Recall@k = 상위 k개 중 hit 수 / 정답 수.

    Args:
        recommended: 순위순 추천 리스트.
        relevant: 정답 집합.
        k: 자를 순위.

    Returns:
        0~1 사이 recall.

    Raises:
        ValueError: 정답 집합이 비었거나 k가 1 미만일 때.
    """
    _check(relevant, k)
    return sum(_hit_flags(recommended, relevant, k)) / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """nDCG@k (binary relevance). DCG / 이상적인 DCG.

    Args:
        recommended: 순위순 추천 리스트.
        relevant: 정답 집합.
        k: 자를 순위.

    Returns:
        0~1 사이 nDCG.

    Raises:
        ValueError: 정답 집합이 비었거나 k가 1 미만일 때.
    """
    _check(relevant, k)
    flags = _hit_flags(recommended, relevant, k)
    dcg = sum(1 / math.log2(rank + 2) for rank, hit in enumerate(flags) if hit)
    idcg = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / idcg


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """AP@k = hit가 난 순위의 precision 합 / min(정답 수, k).

    Args:
        recommended: 순위순 추천 리스트.
        relevant: 정답 집합.
        k: 자를 순위.

    Returns:
        0~1 사이 AP. MAP@k는 유저별 AP의 평균.

    Raises:
        ValueError: 정답 집합이 비었거나 k가 1 미만일 때.
    """
    _check(relevant, k)
    hits = 0
    score = 0.0
    for rank, hit in enumerate(_hit_flags(recommended, relevant, k), start=1):
        if hit:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def _item_kind(item: object) -> str:
    """article_id 비교용 타입 이름. 정수 계열(int, numpy int32 등)은 모두 "int".

    Args:
        item: article_id 하나.

    Returns:
        타입 이름.
    """
    if isinstance(item, numbers.Integral) and not isinstance(item, bool):
        return "int"
    return type(item).__name__


def _check_item_types(
    recommendations: dict[str, list[int]], ground_truth: dict[str, set[int]]
) -> None:
    """추천과 정답의 article_id 타입이 같은지 표본 하나씩으로 확인한다.

    타입이 다르면 set 조회가 전부 실패해 조용히 0점이 되므로 미리 막는다.

    Args:
        recommendations: {customer_id: 추천 리스트}.
        ground_truth: {customer_id: 정답 집합}.

    Raises:
        ValueError: 두 타입이 다를 때.
    """
    gt_item = next((i for s in ground_truth.values() for i in s), None)
    rec_item = next((i for r in recommendations.values() for i in r), None)
    if gt_item is None or rec_item is None:
        return
    rec_kind, gt_kind = _item_kind(rec_item), _item_kind(gt_item)
    if rec_kind != gt_kind:
        raise ValueError(
            f"recommended item dtype({type(rec_item).__name__})이 "
            f"ground truth dtype({type(gt_item).__name__})과 다릅니다"
        )


METRIC_FUNCS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
}


def evaluate_users(
    recommendations: dict[str, list[int]],
    ground_truth: dict[str, set[int]],
    k: int,
    metrics: list[str],
) -> dict[str, float]:
    """정답셋 유저 기준으로 지표별 평균 점수를 낸다.

    정답셋에 없는 유저의 추천은 무시한다. 정답셋 유저가 추천에 없으면 빈 리스트로
    보고 0점을 준다(추천을 못 준 것도 성능에 반영). 시작할 때 추천과 정답의
    article_id 타입이 같은지 확인한다.

    Args:
        recommendations: {customer_id: 순위순 추천 article_id 리스트}.
        ground_truth: {customer_id: 실제 구매 article_id 집합}.
        k: 자를 순위.
        metrics: 계산할 지표 이름 (precision, recall, ndcg, map).

    Returns:
        {지표 이름: 유저 평균 점수}.

    Raises:
        ValueError: 모르는 지표 이름이거나, 정답셋이 비었거나, 추천과 정답의
            article_id 타입이 다를 때.
    """
    unknown = set(metrics) - set(METRIC_FUNCS)
    if unknown:
        raise ValueError(f"모르는 지표: {sorted(unknown)}")
    if not ground_truth:
        raise ValueError("ground_truth가 비어 있음")
    _check_item_types(recommendations, ground_truth)
    totals = dict.fromkeys(metrics, 0.0)
    for user, relevant in ground_truth.items():
        recommended = recommendations.get(user, [])
        for name in metrics:
            totals[name] += METRIC_FUNCS[name](recommended, relevant, k)
    return {name: total / len(ground_truth) for name, total in totals.items()}
