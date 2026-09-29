#!/usr/bin/env python3
"""Upload a local folder to Kaggle as a private Dataset.

Usage:
    uv run python tools/upload_to_kaggle.py

Requirements:
    uv add kaggle

Kaggle authentication must already be configured for the Kaggle CLI.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def make_slug(name: str) -> str:
    """Generate a simple Kaggle slug from the display name."""
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        raise ValueError("Tên không tạo được slug hợp lệ.")
    return slug


def check_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Không tìm thấy Kaggle CLI. Cài bằng: uv add kaggle"
        )

    result = subprocess.run(
        ["kaggle", "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Kaggle CLI không chạy được.")


def get_username() -> str:
    """Try Kaggle auth status first; ask manually if it cannot be parsed."""
    result = subprocess.run(
        ["kaggle", "auth", "status"],
        capture_output=True,
        text=True,
    )
    output = f"{result.stdout}\n{result.stderr}"

    patterns = [
        r"username\s*[:=]\s*([A-Za-z0-9_-]+)",
        r"user(?:name)?\s*[:=]\s*([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    return ask("Kaggle username")


def main() -> int:
    print("=" * 60)
    print("Kaggle Uploader")
    print("=" * 60)
    print()

    try:
        check_kaggle_cli()

        raw_path = ask("Đường dẫn thư mục chứa dataset/model")
        local_path = Path(raw_path.strip('"')).expanduser().resolve()

        if not local_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {local_path}")
        if not local_path.is_dir():
            raise ValueError("Tool hiện chỉ hỗ trợ upload thư mục.")

        title = ask("Tên khi lên Kaggle", local_path.name)
        slug = make_slug(title)
        username = get_username()

        # Optional. Leave empty if you do not want to declare a license.
        print("Kiểm tra lience của dataset/model !!")
        license_name = ask("License")

        dataset_id = f"{username}/{slug}"

        print()
        print("-" * 60)
        print("THÔNG TIN UPLOAD")
        print("-" * 60)
        print(f"Local path : {local_path}")
        print(f"Title      : {title}")
        print(f"Slug       : {slug}")
        print(f"Kaggle ID  : {dataset_id}")
        print(f"Visibility : private")
        print(f"License    : {license_name or '(không khai báo)'}")
        print("-" * 60)
        print()

        confirmation = input("Xác nhận upload? [y/N]: ").strip().lower()
        if confirmation not in {"y", "yes"}:
            print("Đã hủy.")
            return 0

        # Stage a temporary copy so the original model/dataset is never
        # modified by the uploader. Kaggle metadata is added only here.
        with tempfile.TemporaryDirectory(prefix="kaggle_upload_") as tmp:
            staging_dir = Path(tmp) / local_path.name
            shutil.copytree(local_path, staging_dir)

            metadata = {
                "title": title,
                "id": dataset_id,
                "licenses": ([{"name": license_name}] if license_name else []),
            }

            (staging_dir / "dataset-metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print("Đang upload lên Kaggle...\n")

            command = [
                "kaggle",
                "datasets",
                "create",
                "-p",
                str(staging_dir),
                "--dir-mode",
                "zip",
            ]
            result = subprocess.run(command, capture_output=True, text=True)

            if result.stdout.strip():
                print(result.stdout.strip())

            if result.returncode != 0:
                print("\n❌ Upload thất bại.")
                if result.stderr.strip():
                    print("\nKaggle error:")
                    print(result.stderr.strip())
                return result.returncode

            print("\n✅ Upload thành công.")
            print(f"Kaggle Dataset: {dataset_id}")
            return 0

    except KeyboardInterrupt:
        print("\nĐã hủy.")
        return 130
    except Exception as exc:
        print(f"\n❌ Lỗi: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
