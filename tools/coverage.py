"""Auditar cobertura de CSV reales. No aprueba las mediciones por existir archivos."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from .common import save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=Path('evidence'))
    p.add_argument('--host-limitation',type=Path)
    p.add_argument('--out',type=Path,default=Path('evidence/coverage.json'))
    a=p.parse_args();counts=defaultdict(int);rounds=defaultdict(set);issues=[]
    for file in list(a.evidence.rglob('direct.csv'))+list(a.evidence.rglob('service.csv')):
        with file.open(newline='',encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r['status']!='ok' or r['phase']!='stable':continue
                key=(r['mode'],r['filter'],int(r['mask']),int(r['width']),int(r['height']),int(r['concurrency']))
                counts[key]+=1
                if r.get('round'):rounds[key].add(r['round'])
    host_exc=a.host_limitation and a.host_limitation.is_file() and a.host_limitation.stat().st_size>20
    modes=['container_direct','service']+([] if host_exc else ['host_direct'])
    for mode in modes:
        for fil,k,size in [('media',9,2000),('media',9,4000),('media',9,6000),
                           ('media',3,4000),('media',21,4000),('nitidez',3,4000),('laplaciano',3,4000)]:
            key=(mode,fil,k,size,size,1)
            if counts[key]<5:issues.append(dict(configuration=key,complete_repetitions=counts[key],required=5))
    for c in (1,2):
        key=('service','media',9,4000,4000,c)
        if len(rounds[key])<5:issues.append(dict(configuration=key,complete_rounds=len(rounds[key]),required=5))
    save_json(a.out,dict(measurement_coverage_satisfied=not issues,host_limitation=bool(host_exc),pending=issues,
        note='Cuenta series completas; revisar tambien hash/GPU/entorno, validacion y limitaciones. '
             'Este auditor no evalua automaticamente la alternativa menor documentada ni firmas/informe.'))
    print(a.out,'cobertura mediciones:',not issues)
    if issues:raise SystemExit(2)

if __name__=='__main__':main()
