import json
from pathlib import Path
from typing import Any

DATASET_SPLITS = {
    "BKEE": ("train", "valid", "dev", "test"),
    "PHEE": ("train", "dev", "val", "test"),
    "VHE": ("train", "dev", "val", "test"),
    "GENEVA": ("train", "dev", "val", "test"),
}


def read_first_sample(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        first_line = stream.readline()
        remainder = stream.read()
    try:
        payload = json.loads(first_line + remainder)
    except json.JSONDecodeError:
        payload = json.loads(first_line)

    if isinstance(payload, list) and payload:
        return payload[0]
    if isinstance(payload, dict):
        records = payload.get("data")
        if isinstance(records, list) and records:
            return records[0]
        return payload
    return None


def load_first_available_sample(
    project_root: Path, dataset: str | None = None
) -> tuple[Path | None, Any]:
    for name, splits in DATASET_SPLITS.items():
        if dataset is not None and name != dataset:
            continue
        for split in splits:
            path = project_root / "data/raw" / name / f"{split}.json"
            if path.exists():
                sample = read_first_sample(path)
                if sample is not None:
                    return path, sample
    return None, None
