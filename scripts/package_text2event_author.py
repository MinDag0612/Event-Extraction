"""Package only pinned author sources, GENEVA data and native runtime launchers."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(root,data,output):
    if output.exists():raise FileExistsError(output)
    lock=json.loads((root/'configs/text2event_author/source-lock.json').read_text())
    files={}
    for name,digest in lock['files'].items():
        path=root/'third_party/Text2Event-author'/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Author source differs')
        files['author/'+name]=path
    for path in (root/'configs/text2event_author').glob('*'):
        if path.is_file():files['configs/'+path.name]=path
    for name in ['run_text2event_author.py','prepare_text2event_author.py']:
        files['scripts/'+name]=root/'scripts'/name
    for path in data.rglob('*'):
        if path.is_file():files['data/'+str(path.relative_to(data))]=path
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,path in files.items():archive.write(path,name)
        archive.writestr('author-runtime-manifest.json',json.dumps(manifest,indent=2))
    print(json.dumps({'files':len(files),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'bytes':output.stat().st_size}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();package(Path.cwd(),a.data,a.output)
