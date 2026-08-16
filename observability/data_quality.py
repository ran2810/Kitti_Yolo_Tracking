
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class QualityReport:
    dataset: str
    records: int
    errors: list[str]
    warnings: list[str]
    distributions: dict[str, dict[str, int]]

    @property
    def passed(self) -> bool:
        return not self.errors


SCENE_FIELDS = {
    "id", "image_path", "label_path", "summary_text", "num_cars",
    "num_pedestrians", "num_cyclists", "max_occlusion", "max_truncation",
}
ERROR_FIELDS = {
    "id", "image_path", "summary_text", "error_type", "class", "iou",
}


def validate_documents(docs: list[dict[str, Any]], dataset: str) -> QualityReport:
    required = SCENE_FIELDS if dataset == "scene" else ERROR_FIELDS
    errors: list[str] = []
    warnings: list[str] = []
    ids = Counter(str(doc.get("id", "")) for doc in docs)

    if not docs:
        errors.append("document collection is empty")

    for index, doc in enumerate(docs):
        missing = sorted(required - doc.keys())
        if missing:
            errors.append(f"record {index}: missing fields {missing}")
        if not str(doc.get("id", "")).isdigit():
            errors.append(f"record {index}: invalid frame id {doc.get('id')!r}")
        if dataset == "scene":
            for field in ("num_cars", "num_pedestrians", "num_cyclists", "max_occlusion"):
                if not isinstance(doc.get(field), int) or doc.get(field, -1) < 0:
                    errors.append(f"record {index}: {field} must be a non-negative integer")
            truncation = doc.get("max_truncation")
            if not isinstance(truncation, (int, float)) or not 0 <= truncation <= 1:
                errors.append(f"record {index}: max_truncation must be within [0, 1]")
        else:
            if doc.get("error_type") not in {"FP", "FN"}:
                errors.append(f"record {index}: error_type must be FP or FN")
            iou = doc.get("iou")
            if not isinstance(iou, (int, float)) or not 0 <= iou <= 1:
                errors.append(f"record {index}: iou must be within [0, 1]")

    if dataset == "scene":
        duplicate_ids = [frame_id for frame_id, count in ids.items() if count > 1]
        if duplicate_ids:
            errors.append(f"duplicate scene ids: {duplicate_ids[:10]}")
    elif len(ids) == 1 and docs:
        warnings.append("all error documents belong to one frame")

    if dataset == "error" and docs:
        queryable_classes = {"Car", "Pedestrian", "Cyclist"}
        excluded = sum(doc.get("class") not in queryable_classes for doc in docs)
        if excluded:
            warnings.append(
                f"{excluded} error documents ({excluded / len(docs):.1%}) use classes "
                "outside the queryable Car/Pedestrian/Cyclist scope"
            )

    distributions = {}
    for field in (("max_occlusion",) if dataset == "scene" else ("error_type", "class")):
        distributions[field] = dict(Counter(str(doc.get(field)) for doc in docs))

    return QualityReport(dataset, len(docs), errors, warnings, distributions)


def validate_index_count(index_path: Path, expected: int) -> str | None:
    try:
        import faiss
    except ImportError:
        return "FAISS unavailable; index cardinality check skipped"
    index = faiss.read_index(str(index_path))
    return None if index.ntotal == expected else f"index has {index.ntotal} vectors; expected {expected}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/observability/data_quality.json"))
    args = parser.parse_args()

    specs = (
        ("scene", args.data_dir / "kitti_docs.json", args.data_dir / "kitti_index.faiss"),
        ("error", args.data_dir / "error_docs.json", args.data_dir / "error_index.faiss"),
    )
    reports = []
    for dataset, docs_path, index_path in specs:
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        report = validate_documents(docs, dataset)
        index_issue = validate_index_count(index_path, len(docs))
        if index_issue:
            (report.warnings if "skipped" in index_issue else report.errors).append(index_issue)
        reports.append(report)

    payload = {"passed": all(r.passed for r in reports), "reports": [asdict(r) | {"passed": r.passed} for r in reports]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
