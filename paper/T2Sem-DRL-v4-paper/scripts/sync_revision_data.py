"""Sync revision_v43 CSV from t2sem-mec-v2 into paper data/ for plotting."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

CODE_ROOT = Path(r"C:/Users/12293/Desktop/cpscources/cps3962/project/t2sem-mec-v2")
PAPER_ROOT = Path(__file__).resolve().parents[1]
REV_DIR = CODE_ROOT / "results" / "revision_v43"
DATA = PAPER_ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-dir", type=Path, default=REV_DIR)
    args = parser.parse_args()
    rev = args.revision_dir
    DATA.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for name in ("baseline_v43.csv", "revision_summary.csv"):
        src = rev / name
        if src.is_file():
            shutil.copy2(src, DATA / name)
            copied.append(name)

    summary = rev / "revision_summary.csv"
    if summary.is_file():
        shutil.copy2(summary, DATA / "experiment_summary.csv")
        copied.append("experiment_summary.csv (from revision_summary)")

    if not copied:
        raise SystemExit(f"No CSV found under {rev}. Run revision_run.py first.")

    print("Copied to", DATA)
    for c in copied:
        print(" ", c)
    print("\nNext: python scripts/plot_paper_figures.py")


if __name__ == "__main__":
    main()
