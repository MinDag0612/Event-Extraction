from pathlib import Path
from src.data.unified import read_rows

DATASET_SPLITS = {
    'BKEE': ('train', 'dev', 'test'),
    'GENEVA': ('train', 'val', 'test'),
    'PHEE': ('train', 'dev', 'test'),
    'VHE': ('event',),
    'MAVEN-Arg': ('train', 'valid', 'test'),
    'RAMS': ('train', 'dev', 'test'),
}


def load_first_available_sample(project_root: Path, dataset=None):
    for name, splits in DATASET_SPLITS.items():
        if dataset is not None and name != dataset:
            continue
        for split in splits:
            for suffix in ('.json', '.jsonl', '.jsonlines'):
                path = project_root / 'data/raw' / name / (split + suffix)
                if path.exists():
                    rows = read_rows(path)
                    if rows:
                        return path, rows[0]
    return None, None
