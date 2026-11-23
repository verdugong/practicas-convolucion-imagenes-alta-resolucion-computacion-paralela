"""Restaurar los mismos bytes PGM desde gzip; no reconvierte ni redimensiona."""
import gzip
import json
from pathlib import Path
import shutil
from .common import ROOT,sha


def main():
    manifest=json.loads((ROOT/'docs/data_practica3.json').read_text(encoding='utf-8'))
    for entry in manifest['inputs']:
        dest=ROOT/entry['path'];compressed=Path(str(dest)+'.gz')
        if dest.is_file():
            if sha(dest)!=entry['sha256']:raise RuntimeError('PGM existente difiere: '+str(dest))
            print('Verificado existente:',dest.name);continue
        if not compressed.is_file():raise FileNotFoundError(compressed)
        temp=dest.with_suffix(dest.suffix+'.tmp')
        try:
            with gzip.open(compressed,'rb') as f,temp.open('wb') as out:shutil.copyfileobj(f,out)
            if sha(temp)!=entry['sha256']:raise RuntimeError('Hash descomprimido no coincide: '+str(dest))
            temp.replace(dest)
        finally:temp.unlink(missing_ok=True)
        print('Restaurado sin cambios:',dest.name)

if __name__=='__main__':main()
