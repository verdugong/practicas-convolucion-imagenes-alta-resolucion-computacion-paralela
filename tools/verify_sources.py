"""Comprueba que kernels, clases PyCUDA y referencia CPU no cambiaron desde el ZIP."""
import json
from .common import ROOT, sha


def main():
    m=json.loads((ROOT/'docs/origin_practica2.json').read_text(encoding='utf-8'))
    errors=[]
    for rel,expected in m['preserved_sources'].items():
        if sha(ROOT/rel)!=expected:errors.append(rel)
    for entry in m['inputs']:
        if sha(ROOT/entry['path'])!=entry['sha256']:errors.append(entry['path'])
    if errors:raise SystemExit('Difieren fuentes/datos: '+', '.join(errors))
    print('Fuentes originales e imagenes verificadas:',len(m['preserved_sources']),len(m['inputs']))

if __name__=='__main__':main()
