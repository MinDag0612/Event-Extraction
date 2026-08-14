from pprint import pprint
import json

with open("data\\raw\\MAVEN-Arg\\train.jsonl", "r", encoding="utf-8") as f:
    sample = json.loads(next(f))

pprint(sample)

# pprint(sample["events"][10]["argument"].keys())
