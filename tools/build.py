"""Construir desde Git limpio y registrar commit, log e identificador Docker reales."""
import argparse
from pathlib import Path
import subprocess
from .common import ROOT, command, save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='practica3:1.0')
    p.add_argument('--out',type=Path,default=Path('evidence/build'))
    a=p.parse_args()
    revision=command(['git','rev-parse','HEAD']);status=command(['git','status','--porcelain','--untracked-files=no'])
    if revision.get('returncode')!=0:
        p.error('Primero suba/clone o cree el repositorio Git y haga commit de practica 03.')
    if status.get('stdout'):
        p.error('Hay cambios rastreados sin commit. Confirmarlos o usar una copia limpia antes de construir.')
    # Untracked source would be copied by Docker but absent from commit. Evidence is allowed.
    extras=command(['git','ls-files','--others','--exclude-standard']).get('stdout','').splitlines()
    bad=[s for s in extras if not s.startswith(('evidence/','runtime/'))]
    if bad:p.error('Hay archivos sin rastrear fuera de evidence/runtime: '+', '.join(bad[:8]))
    commit=revision['stdout'];a.out.mkdir(parents=True,exist_ok=False)
    cmd=['docker','build','--pull','--build-arg','CODE_COMMIT='+commit,'-t',a.image,'.']
    with (a.out/'build.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
        for line in proc.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=proc.wait()
    save_json(a.out/'image.json',dict(commit=commit,command=cmd,exit_code=code,
        image=command(['docker','image','inspect',a.image]),
        base=command(['docker','image','inspect','nvidia/cuda:12.6.3-devel-ubuntu22.04']),
        docker=command(['docker','version'])))
    if code:raise SystemExit(code)
    print('Construida',a.image,'desde',commit)

if __name__=='__main__':main()
