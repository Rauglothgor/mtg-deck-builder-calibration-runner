"""Merge calibration shard markdown reports into combined metrics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from deckbuilder.experiment.metrics import compute_calibration

CASE_RE = re.compile(
    r"^\| `(?P<deck_id>[^`]+)` \| "
    r"(?P<predicted>[0-9.]+) \| "
    r"(?P<actual>[0-9.]+) \| "
    r"(?:(?P<bias>-?[0-9.]+) \| )?"
    r"(?P<deviation>[0-9.]+) \|$"
)


def parse_cases(report_path: Path) -> list[tuple[float, float]]:
    """Parse predicted/actual pairs from one shard report."""
    pairs: list[tuple[float, float]] = []
    for line in report_path.read_text(encoding="utf-8").splitlines():
        match = CASE_RE.match(line)
        if match is None:
            continue
        pairs.append((float(match.group("predicted")), float(match.group("actual"))))
    return pairs


def main() -> None:
    """Print combined calibration metrics for shard markdown reports."""
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_dir", type=Path)
    args = parser.parse_args()

    report_paths = sorted(args.reports_dir.glob("**/v0_5_calibration_shard_*.md"))
    pairs: list[tuple[float, float]] = []
    for report_path in report_paths:
        report_pairs = parse_cases(report_path)
        print(f"{report_path}: {len(report_pairs)} cases")
        pairs.extend(report_pairs)

    if not pairs:
        raise SystemExit("No shard cases found")

    calibration = compute_calibration(pairs)
    print()
    print(f"cases={len(pairs)}")
    print(f"mean_absolute_deviation={calibration.mean_absolute_deviation:.6f}")
    print(f"max_deviation={calibration.max_deviation:.6f}")
    print(f"mean_calibration_bias={calibration.mean_calibration_bias:.6f}")
    print(f"overconfidence_rate_20={calibration.overconfidence_rate_20:.6f}")
    print(f"overconfidence_rate_30={calibration.overconfidence_rate_30:.6f}")
    print(f"brier_score={calibration.brier_score:.6f}")
    print(f"adversarial_rate={calibration.adversarial_rate:.6f}")
    print(f"decision={calibration.decision}")


if __name__ == "__main__":
    main()
