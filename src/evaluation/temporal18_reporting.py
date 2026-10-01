"""Incremental records and complete/limited result summaries for temporal18."""

import csv
import json
import math
from pathlib import Path

import numpy as np

from .temporal18_metrics import VIEW_GROUPS


METRICS = ("psnr", "ssim", "lpips", "pcc")


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def deduplicate_padding(records):
    # Index position is the identity: repeated tokens in the source list are kept.
    unique = {}
    for row in records:
        unique.setdefault((row["sample_index"], row["view_group"], row["method"]), row)
    return sorted(unique.values(), key=lambda row: (row["sample_index"], row["method"], list(VIEW_GROUPS).index(row["view_group"])))


def summarize(records, metadata):
    records = deduplicate_padding(records)
    selected = [row for row in records if row["method"] == "probabilistic"]
    indices = {row["sample_index"] for row in selected}
    expected = metadata["expected_bins"]
    complete = indices == set(range(expected)) and len(selected) == 3 * expected
    groups = {}
    methods = sorted({row["method"] for row in records}) or ["probabilistic"]
    for method in methods:
        groups[method] = {}
        for group in VIEW_GROUPS:
            rows = [row for row in records if row["method"] == method and row["view_group"] == group]
            scores, undefined = {}, {}
            for name in METRICS:
                values = [row[name] for row in rows]
                undefined[name] = sum(value is None or math.isnan(value) for value in values)
                # Never drop a sample to produce an apparently valid mean.
                scores[name] = None if not values or undefined[name] else float(np.mean(values))
            groups[method][group] = {"num_bins": len(rows), **scores, "undefined_counts": undefined}
    return {**metadata, "complete": complete, "limited": not complete, "processed_bins": len(indices),
            "record_count": len(records), "groups": groups["probabilistic"], "methods": groups}


class EvaluationWriter:
    def __init__(self, directory, metadata, rank=0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata, self.rank = metadata, rank
        self.records = []
        self.journal = self.directory / f"per_bin_metrics.rank{rank:04d}.jsonl"
        self.journal.write_text("")
        if rank == 0:
            write_json(self.directory / "evaluation_summary.json", {**metadata, "complete": False, "processed_bins": 0})

    def append(self, rows):
        self.records.extend(rows)
        with self.journal.open("a") as handle:
            for row in rows:
                handle.write(json.dumps(json_value(row), ensure_ascii=False, allow_nan=False) + "\n")

    def finish(self, records=None):
        records = deduplicate_padding(self.records if records is None else records)
        summary = summarize(records, self.metadata)
        fields = ["sample_index", "bin_token", "scene_id", "stage", "method", "view_group", "pixel_protocol", *METRICS, "metric_issues"]
        with (self.directory / "per_bin_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in records:
                clean = json_value(row)
                clean["metric_issues"] = json.dumps(clean["metric_issues"], ensure_ascii=False)
                writer.writerow(clean)
        write_json(self.directory / "evaluation_summary.json", summary)
        write_json(self.directory / "scores_all_avg.json", {name: summary["groups"]["all_18"][name] for name in METRICS})
        for name in METRICS:
            write_json(self.directory / f"scores_{name}_all.json", [row[name] for row in records
                       if row["view_group"] == "all_18" and row["method"] == "probabilistic"])
        return summary
