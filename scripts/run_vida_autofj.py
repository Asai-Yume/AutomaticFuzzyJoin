#!/usr/bin/env python3
"""Run AutoFJ over converted VIDA datasets.

Default input:
    data/vida_autofj/<family>/<dataset>/
        left.csv
        right.csv
        gt.csv

Default output:
    outputs/vida_autofj_kbwt_ss/<family>/<dataset>/
        predictions.csv
        metrics.csv
        selected_config.json
        run.log
        SUCCESS

The runner processes datasets sequentially and resumes by skipping output
directories that already contain both SUCCESS and a usable metrics.csv.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, TextIO

import pandas as pd


class RunError(RuntimeError):
    """Raised when a converted dataset or AutoFJ result is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AutoFJ over one or more dataset-family folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data") / "vida_autofj",
        help="Root containing dataset-family folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "vida_autofj_kbwt_ss",
        help="Root for experiment outputs.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=("kbwt", "ss"),
        help=(
            "Family folder names under --input-root. Examples: "
            "kbwt ss wt autofj autofj_overlap"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help=(
            "Optional dataset paths relative to each family folder. "
            "For example: --families ss --datasets FF-bikes"
        ),
    )
    parser.add_argument(
        "--precision-target",
        type=float,
        default=0.9,
        help="AutoFJ requested precision target.",
    )
    parser.add_argument(
        "--join-function-space",
        choices=("autofj_sm", "autofj_md", "autofj_lg"),
        default="autofj_sm",
        help="Built-in AutoFJ join-function space.",
    )
    parser.add_argument(
        "--distance-threshold-space",
        type=int,
        default=50,
        help="Number of candidate distance thresholds.",
    )
    parser.add_argument(
        "--column-weight-space",
        type=int,
        default=10,
        help="Number of candidate column weights.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="CPU workers; -1 lets AutoFJ use all processors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace completed and incomplete prior outputs.",
    )
    return parser.parse_args()


def add_local_src_to_path() -> None:
    """Prefer this checkout's src/autofj package without modifying AutoFJ."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RunError(
            f"{label} is missing columns {missing!r}; "
            f"available={list(frame.columns)!r}"
        )


def read_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read converted VIDA or native AutoFJ-format CSV files."""
    required = {
        "left": dataset_dir / "left.csv",
        "right": dataset_dir / "right.csv",
        "ground truth": dataset_dir / "gt.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RunError(f"Missing input files: {missing!r}")

    try:
        left = pd.read_csv(required["left"], encoding="utf-8-sig")
        right = pd.read_csv(required["right"], encoding="utf-8-sig")
        gt = pd.read_csv(required["ground truth"], encoding="utf-8-sig")
    except Exception as exc:
        raise RunError(f"Could not read CSV files: {exc}") from exc

    require_columns(left, ["id"], "left.csv")
    require_columns(right, ["id"], "right.csv")
    require_columns(gt, ["id_l", "id_r"], "gt.csv")

    shared_join_columns = sorted(
        (set(left.columns) & set(right.columns)) - {"id"}
    )
    if not shared_join_columns:
        raise RunError(
            "left.csv and right.csv must share at least one non-ID join column; "
            f"left_columns={list(left.columns)!r}, "
            f"right_columns={list(right.columns)!r}"
        )

    if left["id"].isna().any():
        raise RunError("left.csv contains missing IDs")
    if right["id"].isna().any():
        raise RunError("right.csv contains missing IDs")
    if gt[["id_l", "id_r"]].isna().any().any():
        raise RunError("gt.csv contains missing ID-pair values")

    if left["id"].duplicated().any():
        raise RunError("left.csv contains duplicate IDs")
    if right["id"].duplicated().any():
        raise RunError("right.csv contains duplicate IDs")
    if gt[["id_l", "id_r"]].duplicated().any():
        raise RunError("gt.csv contains duplicate ID pairs")

    left_ids = set(left["id"].astype(str))
    right_ids = set(right["id"].astype(str))
    unknown_left = set(gt["id_l"].astype(str)) - left_ids
    unknown_right = set(gt["id_r"].astype(str)) - right_ids
    if unknown_left or unknown_right:
        raise RunError(
            "gt.csv references unknown IDs: "
            f"left={sorted(unknown_left)[:10]!r}, "
            f"right={sorted(unknown_right)[:10]!r}"
        )

    return left, right, gt


def discover_datasets(family_root: Path) -> list[Path]:
    if not family_root.is_dir():
        return []

    datasets: list[Path] = []
    candidates = [family_root, *family_root.rglob("*")]
    for directory in candidates:
        if not directory.is_dir():
            continue
        if all((directory / filename).is_file()
               for filename in ("left.csv", "right.csv", "gt.csv")):
            datasets.append(directory)
    return sorted(datasets, key=lambda path: str(path).lower())


class Tee:
    """Write text to both the terminal and a log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return repr(value)


def evaluate_pairs(
    predictions: pd.DataFrame,
    gt: pd.DataFrame,
) -> dict[str, Any]:
    require_columns(predictions, ["id_l", "id_r"], "AutoFJ predictions")

    pred_pairs = {
        (str(left_id), str(right_id))
        for left_id, right_id in predictions[["id_l", "id_r"]].itertuples(
            index=False, name=None
        )
    }
    gt_pairs = {
        (str(left_id), str(right_id))
        for left_id, right_id in gt[["id_l", "id_r"]].itertuples(
            index=False, name=None
        )
    }

    true_pairs = pred_pairs & gt_pairs
    false_positive_pairs = pred_pairs - gt_pairs
    false_negative_pairs = gt_pairs - pred_pairs

    precision = len(true_pairs) / len(pred_pairs) if pred_pairs else 0.0
    recall = len(true_pairs) / len(gt_pairs) if gt_pairs else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "predicted_pairs": len(pred_pairs),
        "ground_truth_pairs": len(gt_pairs),
        "true_positives": len(true_pairs),
        "false_positives": len(false_positive_pairs),
        "false_negatives": len(false_negative_pairs),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def usable_completed_output(output_dir: Path) -> bool:
    marker = output_dir / "SUCCESS"
    metrics_path = output_dir / "metrics.csv"
    if not marker.is_file() or not metrics_path.is_file():
        return False
    try:
        metrics = pd.read_csv(metrics_path)
    except Exception:
        return False
    return len(metrics) == 1 and metrics.iloc[0].get("status") == "success"


def write_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    pd.DataFrame([metrics]).to_csv(
        output_dir / "metrics.csv",
        index=False,
        encoding="utf-8",
    )


def run_dataset(
    family: str,
    dataset_name: str,
    dataset_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    if usable_completed_output(output_dir) and not args.overwrite:
        existing = pd.read_csv(output_dir / "metrics.csv").iloc[0].to_dict()
        return "skipped", existing

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            # Incomplete prior run: remove it so the dataset can resume cleanly.
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_metrics: dict[str, Any] = {
        "status": "failed",
        "family": family,
        "dataset": dataset_name,
        "seed": pd.NA,
        "precision_target": args.precision_target,
        "join_function_space": args.join_function_space,
        "distance_threshold_space": args.distance_threshold_space,
        "column_weight_space": args.column_weight_space,
        "n_jobs": args.n_jobs,
        "num_model_calls": 0,
        "num_llm_calls": 0,
    }

    log_path = output_dir / "run.log"
    started = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)

        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            try:
                print("=" * 72)
                print(f"Running {family}/{dataset_name}")
                print(f"Input:  {dataset_dir}")
                print(f"Output: {output_dir}")
                print("=" * 72)

                left, right, gt = read_dataset(dataset_dir)
                print(
                    f"Rows: left={len(left)}, right={len(right)}, "
                    f"gt_pairs={len(gt)}"
                )

                add_local_src_to_path()
                from autofj import AutoFJ  # imported after adding local src

                autofj = AutoFJ(
                    precision_target=args.precision_target,
                    join_function_space=args.join_function_space,
                    distance_threshold_space=args.distance_threshold_space,
                    column_weight_space=args.column_weight_space,
                    n_jobs=args.n_jobs,
                    verbose=True,
                )

                # Do not pass on=["value"]. The current implementation needs its
                # internal autofj_id column retained in the working tables.
                result = autofj.join(left, right, id_column="id")

                require_columns(result, ["id_l", "id_r"], "AutoFJ result")
                predictions = (
                    result[["id_l", "id_r"]]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                predictions.to_csv(
                    output_dir / "predictions.csv",
                    index=False,
                    encoding="utf-8",
                )

                evaluation = evaluate_pairs(predictions, gt)
                runtime = time.perf_counter() - started

                metrics = {
                    **base_metrics,
                    "status": "success",
                    "left_rows": len(left),
                    "right_rows": len(right),
                    **evaluation,
                    "runtime_seconds": runtime,
                }
                write_metrics(output_dir, metrics)

                config = {
                    "selected_column_weights": json_safe(
                        getattr(autofj, "selected_column_weights", None)
                    ),
                    "selected_join_configs": json_safe(
                        getattr(autofj, "selected_join_configs", None)
                    ),
                }
                (output_dir / "selected_config.json").write_text(
                    json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                (output_dir / "SUCCESS").write_text(
                    "success\n",
                    encoding="utf-8",
                )

                print(
                    "Metrics: "
                    f"P={metrics['precision']:.6f} "
                    f"R={metrics['recall']:.6f} "
                    f"F1={metrics['f1']:.6f} "
                    f"time={runtime:.2f}s"
                )
                print("Status: SUCCESS")
                return "converted", metrics

            except Exception as exc:
                runtime = time.perf_counter() - started
                error_text = f"{type(exc).__name__}: {exc}"
                metrics = {
                    **base_metrics,
                    "runtime_seconds": runtime,
                    "error": error_text,
                }
                write_metrics(output_dir, metrics)
                (output_dir / "ERROR.txt").write_text(
                    error_text + "\n\n" + traceback.format_exc(),
                    encoding="utf-8",
                )
                print(f"Status: FAILED — {error_text}")
                return "failed", metrics


def rebuild_summary(output_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if output_root.is_dir():
        for metrics_path in output_root.rglob("metrics.csv"):
            try:
                frame = pd.read_csv(metrics_path)
            except Exception:
                continue
            if len(frame) == 1:
                rows.append(frame)

    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not summary.empty:
        summary = summary.sort_values(
            ["family", "dataset"],
            kind="stable",
        ).reset_index(drop=True)
    summary.to_csv(
        output_root / "all_results.csv",
        index=False,
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()

    if not 0.0 <= args.precision_target <= 1.0:
        print("ERROR: --precision-target must be between 0 and 1.", file=sys.stderr)
        return 2
    if args.distance_threshold_space < 1:
        print("ERROR: --distance-threshold-space must be >= 1.", file=sys.stderr)
        return 2
    if args.column_weight_space < 1:
        print("ERROR: --column-weight-space must be >= 1.", file=sys.stderr)
        return 2

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total_found = 0
    total_run = 0
    total_skipped = 0
    total_failed = 0
    failures: list[str] = []

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Families:    {', '.join(args.families)}")
    print("Seeds:       none (AutoFJ exposes no seed or example sampling)")
    print()

    for family in args.families:
        family_root = input_root / family
        datasets = discover_datasets(family_root)
        if args.datasets:
            requested = {name.replace("\\", "/").strip("/") for name in args.datasets}
            datasets = [
                dataset_dir
                for dataset_dir in datasets
                if dataset_dir.relative_to(family_root).as_posix() in requested
            ]
        total_found += len(datasets)

        family_run = 0
        family_skipped = 0
        family_failed = 0

        print(f"[{family.upper()}] Found {len(datasets)} dataset(s)")

        for dataset_dir in datasets:
            relative = dataset_dir.relative_to(family_root)
            dataset_name = relative.as_posix()
            output_dir = output_root / family / relative

            status, metrics = run_dataset(
                family=family,
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                args=args,
            )

            if status == "converted":
                family_run += 1
                total_run += 1
            elif status == "skipped":
                family_skipped += 1
                total_skipped += 1
                print(f"SKIPPED completed {family}/{dataset_name}")
            else:
                family_failed += 1
                total_failed += 1
                failures.append(
                    f"{family}/{dataset_name}: {metrics.get('error', 'unknown error')}"
                )

        print(
            f"[{family.upper()}] ran {family_run}; "
            f"skipped {family_skipped}; failed {family_failed}"
        )
        print()

    summary = rebuild_summary(output_root)

    failure_log = output_root / "failures.log"
    failure_log.write_text(
        "\n".join(failures) + ("\n" if failures else ""),
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"Datasets found:   {total_found}")
    print(f"Datasets run:     {total_run}")
    print(f"Datasets skipped: {total_skipped}")
    print(f"Datasets failed:  {total_failed}")
    print(f"Results CSV:      {output_root / 'all_results.csv'}")
    print(f"Failure log:      {failure_log}")

    successful_rows = (
        int((summary["status"] == "success").sum())
        if not summary.empty and "status" in summary.columns
        else 0
    )
    print(f"Successful result rows available: {successful_rows}")

    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
