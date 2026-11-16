#!/usr/bin/env python3
"""Evaluate labeled image or video pairs and recommend an EER threshold."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import auc, roc_curve

from face_verify.aggregate import aggregate_similarity
from face_verify.cli import add_pipeline_arguments, verifier_from_args

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class LabeledPair:
    path1: Path
    path2: Path
    label: int
    row_number: int


@dataclass(frozen=True)
class EvaluationMetrics:
    pairs: int
    same_pairs: int
    different_pairs: int
    threshold: float
    accuracy: float
    auc: float
    eer: float
    eer_threshold: float
    far: float
    frr: float


def parse_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "same", "genuine", "true", "positive"}:
        return 1
    if normalized in {"0", "different", "impostor", "false", "negative"}:
        return 0
    raise ValueError(f"Invalid label {value!r}; use 1/0 or same/different.")


def read_pairs(csv_path: str | Path) -> list[LabeledPair]:
    """Read paths relative to the CSV file, which makes datasets portable."""

    source = Path(csv_path)
    if not source.is_file():
        raise ValueError(f"CSV file does not exist: {source}")
    pairs: list[LabeledPair] = []
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"path1", "path2", "label"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("CSV must contain the header: path1,path2,label")
        for row_number, row in enumerate(reader, start=2):
            try:
                first = Path(row["path1"].strip()).expanduser()
                second = Path(row["path2"].strip()).expanduser()
                if not first.is_absolute():
                    first = source.parent / first
                if not second.is_absolute():
                    second = source.parent / second
                pairs.append(
                    LabeledPair(first, second, parse_label(row["label"]), row_number)
                )
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"Invalid CSV row {row_number}: {exc}") from exc
    if not pairs:
        raise ValueError("The CSV contains no labeled pairs.")
    return pairs


def compute_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[EvaluationMetrics, np.ndarray, np.ndarray, np.ndarray]:
    """Compute threshold metrics and the ROC coordinates."""

    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between -1 and 1.")
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
        raise ValueError("Labels and scores must be equally sized, non-empty 1D arrays.")
    if len(np.unique(labels)) < 2:
        raise ValueError("Evaluation needs at least one same and one different pair.")

    predicted = (scores >= threshold).astype(int)
    false_positive = int(np.sum((predicted == 1) & (labels == 0)))
    false_negative = int(np.sum((predicted == 0) & (labels == 1)))
    different_count = int(np.sum(labels == 0))
    same_count = int(np.sum(labels == 1))

    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    # Ignore sklearn's initial infinite threshold. Otherwise perfect separation
    # could incorrectly select that artificial point and report EER = 0.5.
    finite = np.isfinite(roc_thresholds)
    finite_fpr = fpr[finite]
    finite_fnr = 1.0 - tpr[finite]
    finite_thresholds = roc_thresholds[finite]
    eer_index = int(np.nanargmin(np.abs(finite_fpr - finite_fnr)))
    eer = float((finite_fpr[eer_index] + finite_fnr[eer_index]) / 2.0)
    metrics = EvaluationMetrics(
        pairs=len(labels),
        same_pairs=same_count,
        different_pairs=different_count,
        threshold=threshold,
        accuracy=float(np.mean(predicted == labels)),
        auc=float(auc(fpr, tpr)),
        eer=eer,
        eer_threshold=float(finite_thresholds[eer_index]),
        far=false_positive / different_count,
        frr=false_negative / same_count,
    )
    return metrics, fpr, tpr, roc_thresholds


def save_roc_plot(
    output_path: Path,
    fpr: np.ndarray,
    tpr: np.ndarray,
    metrics: EvaluationMetrics,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(fpr, tpr, label=f"ROC (AUC = {metrics.auc:.4f})")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
    axis.scatter(
        [metrics.far],
        [1.0 - metrics.frr],
        label=f"Operating point (threshold = {metrics.threshold:.3f})",
        zorder=3,
    )
    axis.set(xlabel="False acceptance rate (FAR)", ylabel="True acceptance rate (1 - FRR)", title="Face verification ROC")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser, including shared quality options."""

    parser = argparse.ArgumentParser(
        description="Evaluate a path1,path2,label CSV containing image or video pairs."
    )
    parser.add_argument("csv", help="labeled pair CSV")
    parser.add_argument("--strategy", choices=("mean", "pairwise"), default="mean")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--output-dir", default="outputs", help="directory for ROC and metrics")
    add_pipeline_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        pairs = read_pairs(args.csv)
        verifier = verifier_from_args(args)
        scores: list[float] = []
        for index, pair in enumerate(pairs, start=1):
            print(f"[{index}/{len(pairs)}] {pair.path1} <-> {pair.path2}", flush=True)
            try:
                if pair.path1.suffix.lower() in IMAGE_SUFFIXES:
                    first = verifier.extract_image_embeddings(pair.path1)
                else:
                    first = verifier.extract_video_embeddings(pair.path1)
                if pair.path2.suffix.lower() in IMAGE_SUFFIXES:
                    second = verifier.extract_image_embeddings(pair.path2)
                else:
                    second = verifier.extract_video_embeddings(pair.path2)
                scores.append(aggregate_similarity(first, second, args.strategy))
            except (ValueError, OSError) as exc:
                raise ValueError(f"Could not evaluate CSV row {pair.row_number}: {exc}") from exc

        labels = np.asarray([pair.label for pair in pairs], dtype=np.int32)
        score_array = np.asarray(scores, dtype=np.float64)
        metrics, fpr, tpr, _ = compute_metrics(labels, score_array, args.threshold)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_roc_plot(output_dir / "roc_curve.png", fpr, tpr, metrics)
        (output_dir / "metrics.json").write_text(
            json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8"
        )
        np.savetxt(
            output_dir / "scores.csv",
            np.column_stack((labels, score_array)),
            delimiter=",",
            header="label,score",
            comments="",
            fmt=("%d", "%.8f"),
        )
    except (ValueError, OSError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    print(f"\nAccuracy: {metrics.accuracy:.4f}")
    print(f"AUC: {metrics.auc:.4f}")
    print(f"EER: {metrics.eer:.4f}")
    print(f"FAR at threshold {metrics.threshold:.4f}: {metrics.far:.4f}")
    print(f"FRR at threshold {metrics.threshold:.4f}: {metrics.frr:.4f}")
    print(f"Recommended threshold at EER: {metrics.eer_threshold:.6f}")
    print(f"Saved ROC plot and metrics to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
