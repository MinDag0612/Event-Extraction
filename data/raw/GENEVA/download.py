import requests
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/PlusLabNLP/GENEVA/main/data"
OUTPUT_DIR = Path("data/raw/GENEVA")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = ["train.json", "val.json", "test.json"]

for filename in files:
    url = f"{BASE_URL}/{filename}"
    response = requests.get(url)
    response.raise_for_status()

    output_path = OUTPUT_DIR / filename
    output_path.write_bytes(response.content)

    print(f"Downloaded: {filename}")