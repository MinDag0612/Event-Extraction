from pathlib import Path
import json

from src.adapter.auto_adapter import AutoAdapter


def load_first_available_sample(project_root: Path):
    candidates = [
        project_root / "data/raw/BKEE/train.json",
        project_root / "data/raw/BKEE/valid.json",
        project_root / "data/raw/BKEE/dev.json",
        project_root / "data/raw/BKEE/test.json",
        project_root / "data/raw/MAVEN-Arg/train.jsonl",
        project_root / "data/raw/MAVEN-Arg/valid.jsonl",
        project_root / "data/raw/MAVEN-Arg/test.jsonl",
        project_root / "data/raw/RAMS/train.jsonlines",
        project_root / "data/raw/RAMS/dev.jsonlines",
        project_root / "data/raw/RAMS/test.jsonlines",
    ]

    for path in candidates:
        if not path.exists():
            continue

        if path.suffix in {".json", ".jsonl", ".jsonlines"}:
            with path.open("r", encoding="utf-8") as f:
                first_line = f.readline()
                remainder = f.read()
            try:
                payload = json.loads(first_line + remainder)
            except json.JSONDecodeError:
                # Official BKEE files use JSON Lines despite their .json suffix.
                payload = json.loads(first_line)
            if isinstance(payload, list) and payload:
                return path, payload[0]
            if isinstance(payload, dict):
                records = payload.get("data")
                if isinstance(records, list) and records:
                    return path, records[0]
                return path, payload

    return None, None


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    source_path, sample = load_first_available_sample(root)

    if source_path is None:
        print("No dataset sample found under data/raw or data/unified.")
    else:
        adapted = AutoAdapter().adapt(sample)
        print(f"Loaded sample from: {source_path}")
        print(adapted.to_json())
