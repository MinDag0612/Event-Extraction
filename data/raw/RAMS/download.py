from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

import requests


URL = "https://nlp.jhu.edu/rams/RAMS_1.0b.tar.gz"
OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rams-") as temp_dir:
        archive_path = Path(temp_dir) / "RAMS_1.0b.tar.gz"
        with requests.get(URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with archive_path.open("wb") as archive:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    archive.write(chunk)

        extract_dir = Path(temp_dir) / "extracted"
        extract_dir.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(extract_dir, filter="data")

        copied = []
        for filename in ("train.jsonlines", "dev.jsonlines", "test.jsonlines"):
            matches = list(extract_dir.rglob(filename))
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one {filename}, found {len(matches)}")
            shutil.copy2(matches[0], OUTPUT_DIR / filename)
            copied.append(filename)

    print(f"Downloaded RAMS 1.0: {', '.join(copied)}")


if __name__ == "__main__":
    main()
