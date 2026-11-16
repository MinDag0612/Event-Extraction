"""Download a pinned private artifact in resumable HTTP ranges; never log its URL."""
import hashlib
from pathlib import Path
import re
import requests
import zipfile


def download_checked(url, destination, expected_bytes, *, get=None, range_bytes=64*1024**2, max_attempts=5, emit=print):
    if expected_bytes<=0 or range_bytes<=0 or max_attempts<1:raise ValueError('Invalid download limits')
    get=get or requests.get
    destination=Path(destination);temporary=destination.with_suffix('.partial')
    temporary.parent.mkdir(parents=True,exist_ok=True)
    offset=temporary.stat().st_size if temporary.exists() else 0
    if offset>expected_bytes:raise ValueError('Partial artifact exceeds expected length')
    next_progress=((offset//(256*1024**2))+1)*(256*1024**2)
    while offset<expected_bytes:
        end=min(offset+range_bytes,expected_bytes)-1
        for attempt in range(1,max_attempts+1):
            response=None
            try:
                response=get(url,headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(30,60))
                if response.status_code!=206:
                    raise requests.RequestException('Expected HTTP206, got '+str(response.status_code))
                match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                if not match or tuple(map(int,match.groups()))!=(offset,end,expected_bytes):
                    raise requests.RequestException('Unexpected Content-Range')
                with temporary.open('ab') as handle:
                    for chunk in response.iter_content(8*1024**2):
                        if not chunk:continue
                        if offset+len(chunk)>end+1:raise ValueError('Range body exceeds declared end')
                        handle.write(chunk);offset+=len(chunk)
                        if offset>=next_progress:
                            emit('PINNED_DOWNLOAD_BYTES',offset,flush=True);next_progress+=256*1024**2
                if offset!=end+1:raise requests.RequestException('Incomplete range body')
                break
            except requests.RequestException as exc:
                # Do not interpolate network exception text: it can contain a signed URL.
                emit('PINNED_DOWNLOAD_RETRY',{'attempt':attempt,'offset':offset,'error':type(exc).__name__},flush=True)
                if attempt==max_attempts:raise RuntimeError('Pinned download exhausted retries; partial file retained') from None
                if offset==end+1:break
            finally:
                if response is not None:response.close()
    if temporary.stat().st_size!=expected_bytes or not zipfile.is_zipfile(temporary):
        raise ValueError('Downloaded artifact length/ZIP validation failed')
    with temporary.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    temporary.replace(destination)
    emit('PINNED_DOWNLOAD_VERIFIED',expected_bytes,flush=True)
    return {'bytes':expected_bytes,'sha256':digest,'range_bytes':range_bytes}
