from pathlib import Path
import requests

BASE_URL = "https://raw.githubusercontent.com/hoangthuytruc/vhe-dataset/main/event-extraction"

FILES = [
    "event.json",
    "vhe_event_schema.json",
]

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for filename in FILES:
    url = f"{BASE_URL}/{filename}"
    output_path = OUTPUT_DIR / filename

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    output_path.write_bytes(response.content)

    print(f"[OK] {filename}")

print(f"\nDownloaded {len(FILES)} files to: {OUTPUT_DIR}")
