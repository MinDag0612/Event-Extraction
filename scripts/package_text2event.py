"""Create a self-contained Kaggle input archive with code and prepared data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(root,bundle,output):
    if output.exists():raise FileExistsError(output)
    files={}
    for tree in ['src','configs/text2event']:
        for path in (root/tree).rglob('*'):
            if path.is_file() and path.suffix in {'.py','.json'} and '__pycache__' not in path.parts:
                files[str(path.relative_to(root))]=path
    for name in ['train_text2event.py','smoke_text2event.py']:
        files['scripts/'+name]=root/'scripts'/name
    for path in bundle.rglob('*'):
        if path.is_file():files['bundle/'+str(path.relative_to(bundle))]=path
    required=['bundle/report.json','bundle/tokenizer/tokenizer.json']
    for name in ['BKEE','GENEVA','PHEE','VHE']:
        required += [f'bundle/{name}/event.schema']
        required += [f'bundle/{name}/{s}{ext}' for s in ['train','dev','test'] for ext in ['.json','.gold.jsonl']]
    if any(key not in files for key in required):raise ValueError('Missing required bundle artifacts')
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in sorted(files.items())}
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for key,path in sorted(files.items()):archive.write(path,key)
        archive.writestr('package-manifest.json',json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'archive':str(output),'files':len(files),'bytes':output.stat().st_size,
                      'sha256':hashlib.sha256(output.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();package(Path.cwd(),a.bundle,a.output)
