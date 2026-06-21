"""
Create small SecondModel/*_deploy.npy slices for GitHub / Streamlit Cloud.

The full X_test_*.npy files (~180 MB total) exceed GitHub's 100 MB per-file limit.
The dashboard only needs the first 300 rows for LIME/ELI5 and 100 rows for sample pickers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent / "SecondModel"
LIME_ROWS = 300

SLICES: tuple[tuple[str, str, int], ...] = (
    ("X_test_rf_cicids2017.npy", "X_test_rf_cicids2017_deploy.npy", LIME_ROWS),
    ("y_test_rf_cicids2017.npy", "y_test_rf_cicids2017_deploy.npy", LIME_ROWS),
    (
        "X_test_xgb_cicids2017_95percent.npy",
        "X_test_xgb_cicids2017_95percent_deploy.npy",
        LIME_ROWS,
    ),
    (
        "y_test_xgb_cicids2017_95percent.npy",
        "y_test_xgb_cicids2017_95percent_deploy.npy",
        LIME_ROWS,
    ),
)


def slice_array(source: Path, rows: int) -> np.ndarray:
    data = np.load(source)
    if data.shape[0] < rows:
        raise ValueError(
            f"{source.name} has {data.shape[0]} rows; need at least {rows} for deploy slice."
        )
    return np.asarray(data[:rows])


def main() -> int:
    if not BASE.is_dir():
        print(f"Missing folder: {BASE}")
        return 1

    created = 0
    for source_name, deploy_name, rows in SLICES:
        source = BASE / source_name
        deploy = BASE / deploy_name
        if not source.is_file():
            print(f"SKIP (missing source): {source_name}")
            continue
        sliced = slice_array(source, rows)
        np.save(deploy, sliced)
        size_kb = deploy.stat().st_size / 1024
        print(f"Wrote {deploy_name}: shape={sliced.shape}, {size_kb:.1f} KB")
        created += 1

    if created == 0:
        print(
            "No deploy slices created. Train models first so SecondModel/X_test_*.npy exist."
        )
        return 1

    print(f"\nDone. Commit the *_deploy.npy files; keep full X_test_*.npy local only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
