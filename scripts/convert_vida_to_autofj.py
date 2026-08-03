#!/usr/bin/env python3
"""Bulk-convert VIDA KBWT, SS, and WT datasets into AutoFJ inputs.

Expected input layout:
    data/
        kbwt/<dataset>/source.csv, target.csv, ground_truth.csv
        ss/<dataset>/source.csv, target.csv, ground_truth.csv
        wt/<dataset>/source.csv, target.csv, ground_truth.csv

Ground-truth filenames such as ``ground_truth.csv``, ``ground truth.csv``,
``ground-truth.csv``, and ``gt.csv`` are accepted case-insensitively.

Each converted dataset contains exactly:
    left.csv     AutoFJ reference/target table with columns: id,value
    right.csv    AutoFJ source/noisy table with columns: id,value
    gt.csv       Ground-truth ID pairs with columns: id_l,id_r

The converter recursively discovers dataset folders below ``data/kbwt``,
``data/ss``, and ``data/wt`` and mirrors their relative paths under the output
root. AutoFJ expects the reference table on the left and the source/noisy table
on the right.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
import shutil
import sys
import unicodedata
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd


class ConversionError(RuntimeError):
    """Raised when input files cannot be converted safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk-convert VIDA KBWT, SS, and WT datasets for AutoFJ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory containing the kbwt, ss, and wt family folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "vida_autofj",
        help="Root directory for converted AutoFJ datasets.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("kbwt", "ss", "wt"),
        default=("kbwt", "ss", "wt"),
        help="Dataset families to convert.",
    )
    parser.add_argument(
        "--strip-balanced-outer-quotes",
        action="store_true",
        help="Remove one matching pair of outer single or double quotes from match values.",
    )
    parser.add_argument(
        "--strip-whitespace",
        action="store_true",
        help="Strip leading and trailing whitespace from AutoFJ match values.",
    )
    parser.add_argument(
        "--unicode-normalization",
        choices=("none", "NFC", "NFKC"),
        default="none",
        help="Unicode normalization applied to AutoFJ match values.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing converted datasets instead of skipping them.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ConversionError(f"{label} file does not exist: {path}")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise ConversionError(f"Could not read {label} CSV {path}: {exc}") from exc


def require_column(df: pd.DataFrame, column: str, label: str) -> None:
    if column not in df.columns:
        raise ConversionError(
            f"Missing {label} column {column!r}. "
            f"Available columns: {list(df.columns)!r}"
        )


def is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def display_value(value: Any) -> str:
    return "" if is_missing(value) else str(value)


def normalize_match_value(value: Any, args: argparse.Namespace) -> str:
    """Normalize only the value AutoFJ will compare."""
    text = display_value(value)
    if args.strip_whitespace:
        text = text.strip()
    if args.strip_balanced_outer_quotes and len(text) >= 2:
        if text[0] == text[-1] and text[0] in {'"', "'"}:
            text = text[1:-1]
            if args.strip_whitespace:
                text = text.strip()
    if args.unicode_normalization != "none":
        text = unicodedata.normalize(args.unicode_normalization, text)
    return text


def canonical_cell(value: Any) -> tuple[str, str]:
    """Stable exact-lookup key that tolerates CSV numeric inference."""
    if is_missing(value):
        return ("na", "")
    if isinstance(value, (bool, np.bool_)):
        return ("bool", "1" if bool(value) else "0")
    if isinstance(value, (int, np.integer)):
        return ("number", str(int(value)))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return ("number", str(int(number)))
        return ("number", format(number, ".17g"))

    # Remove BOM characters and harmless edge whitespace for GT-to-table ID
    # resolution only. This does not change the value written for AutoFJ.
    text = unicodedata.normalize("NFC", str(value)).replace("\ufeff", "").strip()
    return ("string", text)


def canonical_join_cell(value: Any) -> tuple[str, str]:
    """Lookup key for the designated join column."""
    return canonical_cell(value)


def row_key(row: pd.Series, columns: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple(canonical_cell(row[column]) for column in columns)


def make_ids(prefix: str, count: int) -> list[str]:
    width = max(6, len(str(max(0, count - 1))))
    return [f"{prefix}{index:0{width}d}" for index in range(count)]


def infer_join_columns(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_column: str | None,
    target_column: str | None,
) -> tuple[str, str]:
    if source_column is None:
        shared = [column for column in source.columns if column in target.columns]
        if len(shared) == 1:
            source_column = str(shared[0])
        elif len(source.columns) == 1:
            source_column = str(source.columns[0])
        else:
            raise ConversionError(
                "Could not infer the source join column. "
                f"Source columns={list(source.columns)!r}; shared columns={shared!r}."
            )

    require_column(source, source_column, "source join")

    if target_column is None:
        if source_column in target.columns:
            target_column = source_column
        elif len(target.columns) == 1:
            target_column = str(target.columns[0])
        else:
            raise ConversionError(
                "Could not infer the target join column. "
                f"Target columns={list(target.columns)!r}."
            )

    require_column(target, target_column, "target join")
    return source_column, target_column


def build_reference_value_table(
    target: pd.DataFrame,
    target_column: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Deduplicate KBWT/SS targets by the value AutoFJ will compare."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for raw_value in target[target_column].tolist():
        value = normalize_match_value(raw_value, args)
        if value in seen:
            continue
        seen.add(value)
        rows.append({"value": value})

    ids = make_ids("L", len(rows))
    for index, row in enumerate(rows):
        row["id"] = ids[index]

    left = pd.DataFrame(rows, columns=["id", "value"])
    return left, {row["value"]: row["id"] for row in rows}


def gt_column_for(
    gt: pd.DataFrame,
    side: str,
    table_column: str,
) -> str | None:
    """Return a prefixed GT column first, then an unprefixed column."""
    prefixed = f"{side}-{table_column}"
    if prefixed in gt.columns:
        return prefixed
    if table_column in gt.columns:
        return table_column
    return None


def gt_column_mappings(
    table: pd.DataFrame,
    gt: pd.DataFrame,
    side: str,
) -> list[tuple[str, str]]:
    """Map table columns to prefixed or unprefixed GT columns."""
    mappings: list[tuple[str, str]] = []
    for table_column in table.columns:
        gt_column = gt_column_for(gt, side, str(table_column))
        if gt_column is not None:
            mappings.append((str(table_column), gt_column))
    return mappings


def convert_row_aligned(
    source: pd.DataFrame,
    target: pd.DataFrame,
    gt: pd.DataFrame,
    source_column: str,
    target_column: str,
    gt_source_column: str,
    gt_target_column: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert KBWT/SS row-aligned value-pair ground truth."""
    require_column(gt, gt_source_column, "ground-truth source")
    require_column(gt, gt_target_column, "ground-truth target")

    if len(source) != len(gt):
        raise ConversionError(
            "KBWT/SS conversion requires source.csv and ground truth to have "
            f"the same number of rows; got source={len(source)}, gt={len(gt)}."
        )

    mismatches = [
        index
        for index, (source_value, gt_value) in enumerate(
            zip(source[source_column].tolist(), gt[gt_source_column].tolist())
        )
        if canonical_cell(source_value) != canonical_cell(gt_value)
    ]
    if mismatches:
        raise ConversionError(
            "source.csv and ground truth are not row-aligned. "
            f"First mismatching row indexes: {mismatches[:10]}."
        )

    left, target_value_to_id = build_reference_value_table(
        target, target_column, args
    )

    right_ids = make_ids("R", len(source))
    right = pd.DataFrame(
        {
            "id": right_ids,
            "value": [
                normalize_match_value(value, args)
                for value in source[source_column].tolist()
            ],
        }
    )

    gt_rows: list[dict[str, str]] = []
    for index, gt_target_value in enumerate(gt[gt_target_column].tolist()):
        normalized_target = normalize_match_value(gt_target_value, args)
        left_id = target_value_to_id.get(normalized_target)
        if left_id is None:
            raise ConversionError(
                "Ground truth contains a target value absent from target.csv "
                f"after normalization at row {index}: {gt_target_value!r}"
            )
        gt_rows.append({"id_l": left_id, "id_r": right_ids[index]})

    gt_output = (
        pd.DataFrame(gt_rows, columns=["id_l", "id_r"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return left, right, gt_output


def build_join_index(
    table: pd.DataFrame,
    join_column: str,
) -> dict[tuple[str, str], list[int]]:
    index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row_index, value in enumerate(table[join_column].tolist()):
        index[canonical_join_cell(value)].append(row_index)
    return index


def relaxed_text_key(value: Any) -> str:
    """Comparison key that ignores spacing, punctuation, case, and BOMs.

    This is used only to resolve ground-truth rows to existing input records.
    It does not alter the values written to left.csv or right.csv.
    """
    text = unicodedata.normalize("NFKC", display_value(value))
    text = text.replace("\ufeff", "").casefold()
    return "".join(character for character in text if character.isalnum())


def build_relaxed_join_index(
    table: pd.DataFrame,
    join_column: str,
) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for row_index, value in enumerate(table[join_column].tolist()):
        index[relaxed_text_key(value)].append(row_index)
    return index


def exact_candidates(
    table: pd.DataFrame,
    gt_row: pd.Series,
    mappings: Sequence[tuple[str, str]],
) -> list[int]:
    """Find rows matching every non-missing GT field available for that side."""
    usable = [
        (table_column, gt_column)
        for table_column, gt_column in mappings
        if not is_missing(gt_row[gt_column])
    ]
    if not usable:
        return []

    candidates: list[int] = []
    for row_index, table_row in table.iterrows():
        if all(
            canonical_cell(table_row[table_column])
            == canonical_cell(gt_row[gt_column])
            for table_column, gt_column in usable
        ):
            candidates.append(int(row_index))
    return candidates


def narrow_candidates_with_auxiliary(
    table: pd.DataFrame,
    candidates: list[int],
    mappings: Sequence[tuple[str, str]],
    join_column: str,
    gt_row: pd.Series,
) -> list[int]:
    """Use exact auxiliary fields when they narrow, but never erase candidates."""
    narrowed_candidates = list(candidates)
    for table_column, gt_column in mappings:
        if table_column == join_column or is_missing(gt_row[gt_column]):
            continue
        narrowed = [
            row_index
            for row_index in narrowed_candidates
            if canonical_cell(table.iloc[row_index][table_column])
            == canonical_cell(gt_row[gt_column])
        ]
        if narrowed:
            narrowed_candidates = narrowed
            if len(narrowed_candidates) == 1:
                break
    return narrowed_candidates


def auxiliary_only_candidates(
    table: pd.DataFrame,
    mappings: Sequence[tuple[str, str]],
    join_column: str,
    gt_row: pd.Series,
) -> list[int]:
    """Match all available auxiliary fields while ignoring the join field."""
    auxiliary_mappings = [
        (table_column, gt_column)
        for table_column, gt_column in mappings
        if table_column != join_column and not is_missing(gt_row[gt_column])
    ]
    if not auxiliary_mappings:
        return []
    return exact_candidates(table, gt_row, auxiliary_mappings)


def leading_year_and_remainder(value: Any) -> tuple[int, str] | None:
    """Parse strings like '1859 Viscount Palmerston'."""
    text = unicodedata.normalize("NFKC", display_value(value))
    text = text.replace("\ufeff", "").strip()
    match = re.match(r"^(\d{4})\s+(.+)$", text)
    if match is None:
        return None
    return int(match.group(1)), relaxed_text_key(match.group(2))


def year_aware_candidates(
    table: pd.DataFrame,
    join_column: str,
    gt_value: Any,
) -> list[int]:
    """Resolve a missing year-prefixed value by exact name and nearest year.

    This fallback is deliberately narrow: both the GT value and candidate table
    values must begin with a four-digit year, and the remainder after the year
    must match exactly after punctuation/spacing normalization.
    """
    parsed_gt = leading_year_and_remainder(gt_value)
    if parsed_gt is None:
        return []

    gt_year, gt_remainder = parsed_gt
    year_candidates: list[tuple[int, int]] = []
    for row_index, raw_value in enumerate(table[join_column].tolist()):
        parsed_candidate = leading_year_and_remainder(raw_value)
        if parsed_candidate is None:
            continue
        candidate_year, candidate_remainder = parsed_candidate
        if candidate_remainder == gt_remainder:
            year_candidates.append((row_index, abs(candidate_year - gt_year)))

    if not year_candidates:
        return []

    minimum_difference = min(difference for _, difference in year_candidates)
    return [
        row_index
        for row_index, difference in year_candidates
        if difference == minimum_difference
    ]


def resolve_record_candidates(
    table: pd.DataFrame,
    join_index: dict[tuple[str, str], list[int]],
    relaxed_join_index: dict[str, list[int]],
    mappings: Sequence[tuple[str, str]],
    join_column: str,
    join_gt_column: str,
    gt_row: pd.Series,
    *,
    allow_year_fallback: bool,
) -> tuple[list[int], str]:
    """Resolve one GT side deterministically.

    Resolution order:
      1. Match every available non-missing GT field.
      2. Match the join-column value exactly.
      3. Match the join value while ignoring case, spacing, and punctuation.
      4. Match all available auxiliary fields while ignoring the join field.
      5. For year-prefixed values only, match the exact name remainder and use
         the uniquely nearest leading year.

    Auxiliary fields are used to narrow candidate sets when possible. A field
    that matches no current candidate is ignored because WT metadata can differ
    between ground truth and the original input table.
    """
    full_matches = exact_candidates(table, gt_row, mappings)
    if full_matches:
        return full_matches, "full_record"

    exact_join_candidates = list(
        join_index.get(canonical_join_cell(gt_row[join_gt_column]), [])
    )
    if exact_join_candidates:
        candidates = narrow_candidates_with_auxiliary(
            table,
            exact_join_candidates,
            mappings,
            join_column,
            gt_row,
        )
        method = (
            "unique_join_value"
            if len(candidates) == 1
            else "ambiguous_join_value"
        )
        return candidates, method

    relaxed_join_candidates = list(
        relaxed_join_index.get(relaxed_text_key(gt_row[join_gt_column]), [])
    )
    if relaxed_join_candidates:
        candidates = narrow_candidates_with_auxiliary(
            table,
            relaxed_join_candidates,
            mappings,
            join_column,
            gt_row,
        )
        method = (
            "relaxed_join_value"
            if len(candidates) == 1
            else "ambiguous_relaxed_join_value"
        )
        return candidates, method

    if allow_year_fallback:
        year_candidates = year_aware_candidates(
            table,
            join_column,
            gt_row[join_gt_column],
        )
        if year_candidates:
            candidates = narrow_candidates_with_auxiliary(
                table,
                year_candidates,
                mappings,
                join_column,
                gt_row,
            )
            method = (
                "nearest_year_same_name"
                if len(candidates) == 1
                else "ambiguous_nearest_year_same_name"
            )
            return candidates, method

    auxiliary_candidates = auxiliary_only_candidates(
        table,
        mappings,
        join_column,
        gt_row,
    )
    if auxiliary_candidates:
        method = (
            "auxiliary_record"
            if len(auxiliary_candidates) == 1
            else "ambiguous_auxiliary_record"
        )
        return auxiliary_candidates, method

    return [], "unresolved"


def convert_record_lookup(
    source: pd.DataFrame,
    target: pd.DataFrame,
    gt: pd.DataFrame,
    source_column: str,
    target_column: str,
    gt_source_column: str,
    gt_target_column: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert WT ground truth while preserving unmatched source records.

    The output reference table contains only ``id,value``. Therefore, target
    records that have the same normalized join value are deduplicated: AutoFJ
    could not distinguish separate IDs carrying identical visible values.
    """
    require_column(gt, gt_source_column, "ground-truth source")
    require_column(gt, gt_target_column, "ground-truth target")

    source_mappings = gt_column_mappings(source, gt, "source")
    target_mappings = gt_column_mappings(target, gt, "target")

    if not any(column == source_column for column, _ in source_mappings):
        raise ConversionError(
            f"Could not map source join column {source_column!r} to ground truth. "
            f"Available GT columns: {list(gt.columns)!r}"
        )
    if not any(column == target_column for column, _ in target_mappings):
        raise ConversionError(
            f"Could not map target join column {target_column!r} to ground truth. "
            f"Available GT columns: {list(gt.columns)!r}"
        )

    source_reset = source.reset_index(drop=True)
    target_reset = target.reset_index(drop=True)

    right_ids = make_ids("R", len(source_reset))
    right = pd.DataFrame(
        {
            "id": right_ids,
            "value": [
                normalize_match_value(value, args)
                for value in source_reset[source_column].tolist()
            ],
        }
    )

    # Deduplicate the AutoFJ reference side by the exact value AutoFJ sees.
    ordered_match_values: list[str] = []
    seen_match_values: set[str] = set()
    target_row_match_values: list[str] = []

    for raw_value in target_reset[target_column].tolist():
        match_value = normalize_match_value(raw_value, args)
        target_row_match_values.append(match_value)
        if match_value not in seen_match_values:
            seen_match_values.add(match_value)
            ordered_match_values.append(match_value)

    left_ids = make_ids("L", len(ordered_match_values))
    left = pd.DataFrame(
        {"id": left_ids, "value": ordered_match_values},
        columns=["id", "value"],
    )
    match_value_to_left_id = dict(zip(ordered_match_values, left_ids))
    target_row_to_left_id = [
        match_value_to_left_id[match_value]
        for match_value in target_row_match_values
    ]

    source_join_index = build_join_index(source_reset, source_column)
    source_relaxed_join_index = build_relaxed_join_index(
        source_reset, source_column
    )
    target_join_index = build_join_index(target_reset, target_column)
    target_relaxed_join_index = build_relaxed_join_index(
        target_reset, target_column
    )

    gt_pairs: list[dict[str, str]] = []
    unresolved: list[dict[str, Any]] = []

    for gt_index, gt_row in gt.iterrows():
        source_candidates, source_method = resolve_record_candidates(
            source_reset,
            source_join_index,
            source_relaxed_join_index,
            source_mappings,
            source_column,
            gt_source_column,
            gt_row,
            allow_year_fallback=False,
        )
        target_candidates, target_method = resolve_record_candidates(
            target_reset,
            target_join_index,
            target_relaxed_join_index,
            target_mappings,
            target_column,
            gt_target_column,
            gt_row,
            allow_year_fallback=True,
        )

        target_left_ids = {
            target_row_to_left_id[row_index]
            for row_index in target_candidates
        }

        if (
            not source_candidates
            or not target_candidates
            or len(target_left_ids) != 1
        ):
            unresolved.append(
                {
                    "gt_row": int(gt_index),
                    "source_match_count": len(source_candidates),
                    "target_match_count": len(target_candidates),
                    "distinct_target_left_ids": len(target_left_ids),
                    "source_method": source_method,
                    "target_method": target_method,
                    "source_value": display_value(gt_row[gt_source_column]),
                    "target_value": display_value(gt_row[gt_target_column]),
                }
            )
            continue

        target_id = next(iter(target_left_ids))

        # Duplicate source records are expanded because they are
        # indistinguishable from the available GT fields.
        for source_index in source_candidates:
            gt_pairs.append(
                {"id_l": target_id, "id_r": right_ids[source_index]}
            )

    if unresolved:
        raise ConversionError(
            "Some WT ground-truth rows could not be resolved to input records. "
            f"First unresolved rows: {unresolved[:10]!r}"
        )

    gt_output = (
        pd.DataFrame(gt_pairs, columns=["id_l", "id_r"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return left, right, gt_output

def validate_output(
    left: pd.DataFrame,
    right: pd.DataFrame,
    gt: pd.DataFrame,
) -> None:
    if list(left.columns) != ["id", "value"]:
        raise ConversionError("Converted left.csv must contain exactly id,value.")
    if list(right.columns) != ["id", "value"]:
        raise ConversionError("Converted right.csv must contain exactly id,value.")
    if list(gt.columns) != ["id_l", "id_r"]:
        raise ConversionError("Converted gt.csv must contain exactly id_l,id_r.")

    if left["id"].duplicated().any():
        raise ConversionError("Converted left.csv contains duplicate IDs.")
    if right["id"].duplicated().any():
        raise ConversionError("Converted right.csv contains duplicate IDs.")
    if gt.duplicated().any():
        raise ConversionError("Converted gt.csv contains duplicate pairs.")

    missing_left_ids = set(gt["id_l"]) - set(left["id"])
    missing_right_ids = set(gt["id_r"]) - set(right["id"])
    if missing_left_ids or missing_right_ids:
        raise ConversionError(
            "Converted GT references unknown IDs: "
            f"left={sorted(missing_left_ids)[:10]!r}, "
            f"right={sorted(missing_right_ids)[:10]!r}."
        )


def write_conversion(
    output_dir: Path,
    left: pd.DataFrame,
    right: pd.DataFrame,
    gt: pd.DataFrame,
    overwrite: bool,
) -> str:
    """Write exactly left.csv, right.csv, and gt.csv.

    Direct file writing avoids the Windows directory-renaming PermissionError
    caused by replacing a temporary directory with ``os.replace``.
    """
    output_dir = output_dir.resolve()
    required_outputs = ("left.csv", "right.csv", "gt.csv")

    if output_dir.exists() and not overwrite:
        if output_dir.is_dir() and all(
            (output_dir / filename).is_file() for filename in required_outputs
        ):
            return "skipped"
        raise ConversionError(
            f"Incomplete output already exists: {output_dir}. "
            "Use --overwrite to replace it."
        )

    if output_dir.exists():
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        left.to_csv(output_dir / "left.csv", index=False, encoding="utf-8")
        right.to_csv(output_dir / "right.csv", index=False, encoding="utf-8")
        gt.to_csv(output_dir / "gt.csv", index=False, encoding="utf-8")
        return "converted"
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def canonical_filename(path: Path) -> str:
    return "".join(
        character for character in path.stem.lower() if character.isalnum()
    )


def find_named_csv(directory: Path, expected_name: str) -> Path | None:
    expected = expected_name.lower()
    matches = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and path.name.lower() == expected
    ]
    if len(matches) > 1:
        raise ConversionError(
            f"Multiple files named {expected_name!r} found in {directory}."
        )
    return matches[0] if matches else None


def find_ground_truth_file(directory: Path) -> Path | None:
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and canonical_filename(path) in {"groundtruth", "gt"}
    ]
    if not candidates:
        return None

    preferred_names = (
        "ground_truth.csv",
        "ground truth.csv",
        "ground-truth.csv",
        "gt.csv",
    )
    rank = {name: index for index, name in enumerate(preferred_names)}
    candidates.sort(
        key=lambda path: (
            rank.get(path.name.lower(), 99),
            path.name.lower(),
        )
    )
    return candidates[0]


def find_rows_file(directory: Path) -> Path | None:
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".txt"
        and canonical_filename(path).startswith("rows")
    ]
    return (
        sorted(candidates, key=lambda path: path.name.lower())[0]
        if candidates
        else None
    )


def infer_columns_from_rows_file(
    rows_file: Path | None,
    source: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[str | None, str | None]:
    if rows_file is None:
        return None, None

    try:
        text = rows_file.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = rows_file.read_text(encoding="latin-1")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or ":" not in lines[0]:
        return None, None

    source_column, target_column = (
        part.strip() for part in lines[0].split(":", maxsplit=1)
    )
    if source_column in source.columns and target_column in target.columns:
        return source_column, target_column
    return None, None


def infer_columns_from_ground_truth(
    source: pd.DataFrame,
    target: pd.DataFrame,
    gt: pd.DataFrame,
) -> tuple[str | None, str | None]:
    prefixed_source = [
        str(column)[len("source-") :]
        for column in gt.columns
        if str(column).startswith("source-")
        and str(column)[len("source-") :] in source.columns
    ]
    prefixed_target = [
        str(column)[len("target-") :]
        for column in gt.columns
        if str(column).startswith("target-")
        and str(column)[len("target-") :] in target.columns
    ]

    if len(prefixed_source) == 1 and len(prefixed_target) == 1:
        return prefixed_source[0], prefixed_target[0]

    shared_prefixed = [
        column for column in prefixed_source if column in prefixed_target
    ]
    if len(shared_prefixed) == 1:
        return shared_prefixed[0], shared_prefixed[0]

    # Support unprefixed WT ground truth, such as:
    # last name, first name, Username
    source_unprefixed = [
        str(column) for column in source.columns if column in gt.columns
    ]
    target_unprefixed = [
        str(column) for column in target.columns if column in gt.columns
    ]

    source_only = [
        column for column in source_unprefixed if column not in target.columns
    ]
    target_only = [
        column for column in target_unprefixed if column not in source.columns
    ]
    if len(source_only) == 1 and len(target_only) == 1:
        return source_only[0], target_only[0]

    if len(source.columns) == 1 and source.columns[0] in gt.columns:
        source_hint = str(source.columns[0])
    else:
        source_hint = None
    if len(target.columns) == 1 and target.columns[0] in gt.columns:
        target_hint = str(target.columns[0])
    else:
        target_hint = None

    return source_hint, target_hint


def discover_dataset_directories(family_root: Path) -> list[Path]:
    """Find directories containing both source.csv and target.csv."""
    if not family_root.is_dir():
        return []

    datasets: list[Path] = []
    directories = [family_root, *sorted(family_root.rglob("*"))]
    for directory in directories:
        if not directory.is_dir():
            continue
        if (
            find_named_csv(directory, "source.csv") is not None
            and find_named_csv(directory, "target.csv") is not None
        ):
            datasets.append(directory)
    return datasets


def convert_dataset(
    family: str,
    dataset_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> str:
    source_path = find_named_csv(dataset_dir, "source.csv")
    target_path = find_named_csv(dataset_dir, "target.csv")
    gt_path = find_ground_truth_file(dataset_dir)

    if source_path is None or target_path is None:
        raise ConversionError(
            f"Dataset is missing source.csv or target.csv: {dataset_dir}"
        )
    if gt_path is None:
        raise ConversionError(
            "No supported ground-truth CSV found in "
            f"{dataset_dir}. Expected ground_truth.csv, ground truth.csv, "
            "ground-truth.csv, or gt.csv."
        )

    source = read_csv(source_path, "source")
    target = read_csv(target_path, "target")
    gt = read_csv(gt_path, "ground truth")

    rows_source, rows_target = infer_columns_from_rows_file(
        find_rows_file(dataset_dir), source, target
    )
    gt_source_hint, gt_target_hint = infer_columns_from_ground_truth(
        source, target, gt
    )

    source_column, target_column = infer_join_columns(
        source,
        target,
        rows_source or gt_source_hint,
        rows_target or gt_target_hint,
    )

    gt_source_column = gt_column_for(gt, "source", source_column)
    gt_target_column = gt_column_for(gt, "target", target_column)
    if gt_source_column is None:
        raise ConversionError(
            f"Missing ground-truth source column for {source_column!r}. "
            f"Expected 'source-{source_column}' or '{source_column}'. "
            f"Available columns: {list(gt.columns)!r}"
        )
    if gt_target_column is None:
        raise ConversionError(
            f"Missing ground-truth target column for {target_column!r}. "
            f"Expected 'target-{target_column}' or '{target_column}'. "
            f"Available columns: {list(gt.columns)!r}"
        )

    if family in {"kbwt", "ss"}:
        left, right, gt_output = convert_row_aligned(
            source,
            target,
            gt,
            source_column,
            target_column,
            gt_source_column,
            gt_target_column,
            args,
        )
    else:
        left, right, gt_output = convert_record_lookup(
            source,
            target,
            gt,
            source_column,
            target_column,
            gt_source_column,
            gt_target_column,
            args,
        )

    validate_output(left, right, gt_output)
    return write_conversion(
        output_dir=output_dir,
        left=left,
        right=right,
        gt=gt_output,
        overwrite=args.overwrite,
    )


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    total_found = 0
    total_converted = 0
    total_skipped = 0
    total_failed = 0
    failures: list[tuple[str, str]] = []

    print(f"Input root:  {data_root}")
    print(f"Output root: {output_root}")
    print()

    for family in args.families:
        family_root = data_root / family
        dataset_dirs = discover_dataset_directories(family_root)
        family_found = len(dataset_dirs)
        family_converted = 0
        family_skipped = 0
        family_failed = 0
        total_found += family_found

        print(f"[{family.upper()}] Found {family_found} dataset(s) in {family_root}")

        if not family_root.is_dir():
            print(f"[{family.upper()}] WARNING: folder does not exist")
            print()
            continue

        for dataset_dir in dataset_dirs:
            relative_path = dataset_dir.relative_to(family_root)
            dataset_name = relative_path.as_posix()
            output_dir = output_root / family / relative_path

            try:
                status = convert_dataset(
                    family=family,
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    args=args,
                )
                if status == "converted":
                    family_converted += 1
                    total_converted += 1
                    print(f"  CONVERTED {dataset_name}")
                else:
                    family_skipped += 1
                    total_skipped += 1
                    print(f"  SKIPPED   {dataset_name}")
            except ConversionError as exc:
                family_failed += 1
                total_failed += 1
                failures.append((f"{family}/{dataset_name}", str(exc)))
                print(f"  FAILED    {dataset_name}: {exc}")
            except Exception as exc:
                family_failed += 1
                total_failed += 1
                message = f"{type(exc).__name__}: {exc}"
                failures.append((f"{family}/{dataset_name}", message))
                print(f"  FAILED    {dataset_name}: {message}")

        print(
            f"[{family.upper()}] Converted {family_converted}/{family_found}; "
            f"skipped {family_skipped}; failed {family_failed}"
        )
        print()

    print("=" * 68)
    print(f"Datasets found:     {total_found}")
    print(f"Datasets converted: {total_converted}")
    print(f"Datasets skipped:   {total_skipped}")
    print(f"Datasets failed:    {total_failed}")

    if failures:
        print()
        print("Failures:")
        for dataset_name, message in failures:
            print(f"  {dataset_name}: {message}")

    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
