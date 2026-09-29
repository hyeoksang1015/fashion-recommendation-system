"""ALS / BPR 견고성 확인 CLI 진입점.

역할: ALS와 BPR의 차이가 노이즈인지(paired bootstrap, 시드 반복) 확인하고, hit 유저
    수, hit 평균 순위, 카탈로그 커버리지를 비교한다.
동작: run_robustness 결과를 output_path json과 per_user_path parquet에 저장하고
    표를 로그로 낸다.
사용법: python scripts/run_robustness.py --config configs/robustness.yaml
"""

import argparse
import logging
from src.pipeline.robustness import run_robustness
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def _row(label: str, cells: list[str], width: int = 26) -> str:
    """표 한 줄을 만든다.

    Args:
        label: 첫 열.
        cells: 나머지 열 (이미 서식이 적용된 문자열).
        width: 첫 열 폭.

    Returns:
        한 줄 문자열.
    """
    return f"{label:<{width}}" + "".join(f"{c:>14}" for c in cells)


def _fmt(value: float | int | None) -> str:
    """hit 요약 값을 표 칸 문자열로 만든다.

    Args:
        value: 정수(건수), 실수(평균 순위), None(해당 hit 없음).

    Returns:
        "-", 소수 둘째 자리, 또는 정수 문자열.
    """
    if value is None:
        return "-"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def format_report(out: dict) -> str:
    """run_robustness 결과를 표 문자열로 만든다.

    Args:
        out: run_robustness 결과.

    Returns:
        로그로 낼 표 문자열.
    """
    lines = []
    for subset, by_model in out["results"].items():
        names = list(by_model)
        metrics = [m for m in by_model[names[0]] if m != "n_users"]
        n_users = by_model[names[0]]["n_users"]
        lines += [f"[{subset} 유저: {n_users}명]", _row("", names)]
        lines += [
            _row(m + "@k", [f"{by_model[n][m]:.4f}" for n in names]) for m in metrics
        ]
        lines.append("")

    lines += [
        "[paired bootstrap: BPR - ALS]",
        _row("", ["mean_diff", "ci_low", "ci_high"]),
    ]
    for subset, by_metric in out["bootstrap"].items():
        for m, b in by_metric.items():
            cells = [f"{b[key]:+.5f}" for key in ("mean_diff", "ci_low", "ci_high")]
            lines.append(_row(f"{subset}/{m}@k", cells))
    lines.append("")

    for name, summary in out["seeds"].items():
        metrics = list(summary["mean"])
        lines += [f"[시드 반복: {name}]", _row("seed", [m + "@k" for m in metrics])]
        for run in summary["runs"]:
            lines.append(_row(str(run["seed"]), [f"{run[m]:.4f}" for m in metrics]))
        lines.append(_row("mean", [f"{summary['mean'][m]:.4f}" for m in metrics]))
        lines.append(_row("std", [f"{summary['std'][m]:.5f}" for m in metrics]))
        lines.append("")

    names = list(out["hits"])
    lines += ["[커버 유저 추천만: hit와 카탈로그]", _row("", names)]
    for key in out["hits"][names[0]]:
        lines.append(_row(key, [_fmt(out["hits"][n][key]) for n in names]))
    for key in ("n_distinct_items", "catalog_coverage", "trained_item_coverage"):
        cells = [
            f"{out['catalog'][n][key]:.2%}"
            if "coverage" in key
            else str(out["catalog"][n][key])
            for n in names
        ]
        lines.append(_row(key, cells))
    return "\n".join(lines)


def main() -> None:
    """config 경로를 받아 견고성 확인을 실행한다."""
    parser = argparse.ArgumentParser(description="ALS / BPR 견고성 확인")
    parser.add_argument("--config", default="configs/robustness.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("\n%s", format_report(run_robustness(load_config(args.config))))


if __name__ == "__main__":
    main()
