from pathlib import Path
from urllib.request import urlopen
import json

REVISION = '41489b08675d0b63a1eb7cd4f2d1946952833eb1'
BASE = f'https://raw.githubusercontent.com/ZhaoyueSun/PHEE/{REVISION}'

def main():
    root = Path(__file__).resolve().parent
    for name in ('train.json', 'dev.json', 'test.json', 'LICENSE'):
        remote = f'data/json/{name}' if name.endswith('.json') else name
        with urlopen(f'{BASE}/{remote}', timeout=60) as response:
            payload = response.read()
        if name.endswith('.json'):
            for line in payload.decode('utf-8').splitlines():
                row = json.loads(line)
                assert {'id', 'context', 'annotations'} <= row.keys()
        temporary = root / (name + '.tmp')
        temporary.write_bytes(payload)
        temporary.replace(root / name)
        print(f'Downloaded {name}')
    (root / 'source.json').write_text(json.dumps({'repository':'https://github.com/ZhaoyueSun/PHEE','revision':REVISION,'format':'data/json','version':'original v1'},indent=2))

if __name__ == '__main__':
    main()
