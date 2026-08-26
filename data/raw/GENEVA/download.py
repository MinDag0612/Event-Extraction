import requests
from pathlib import Path
import csv

BASE_URL = "https://raw.githubusercontent.com/PlusLabNLP/GENEVA/main"
OUTPUT_DIR = Path("data/raw/GENEVA")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = [
    # "data/train.json",
    # "data/val.json",
    # "data/test.json",
    "meta_data/fn2geneva_mapping_annotations.tsv",
]

for file in files:
    url = f"{BASE_URL}/{file}"

    response = requests.get(url)
    response.raise_for_status()

    output_path = OUTPUT_DIR / Path(file).name
    output_path.write_bytes(response.content)

    print(f"Downloaded: {file}")
    
    
# Fix fn2geneva_mapping_annotations
print("[FIX] fn2geneva_mapping_annotations")
file_path = OUTPUT_DIR / "fn2geneva_mapping_annotations.tsv"

with open(file_path, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.reader(f, delimiter="\t"))

# Giữ nguyên header
header = rows[0]

# Từ dòng 2 trở đi, xóa cột thứ 6 (index 5)
data = [
    row[:5] + row[6:]
    for row in rows[1:]
]

# Ghi đè
with open(file_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(header)
    writer.writerows(data)

print("Done")