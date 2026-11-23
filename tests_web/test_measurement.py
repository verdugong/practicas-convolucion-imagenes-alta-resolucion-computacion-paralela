"""El resumen no debe publicar mediciones incompletas ni mezclar entradas."""
import csv
from pathlib import Path
from tools.summarize import consolidate
from tools.common import FIELDS


def write(path,count,status='ok',digest='abc'):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
        for i in range(count):w.writerow(dict(mode='service',experiment='resolucion',filter='media',
            width=4000,height=4000,mask=9,threads=256,concurrency=1,phase='stable',
            status=status,sha256_input=digest,kernel_ms=i+1,environment='test'))


def test_incomplete_has_no_statistics(tmp_path):
    p=tmp_path/'a.csv';write(p,4)
    groups,failures,rows=consolidate([p])
    assert not groups[0]['complete'] and 'kernel_ms' not in groups[0]


def test_errors_not_zero_and_hashes_not_mixed(tmp_path):
    a,b,c=[tmp_path/(n+'.csv') for n in 'abc']
    write(a,5);write(b,3,status='error');write(c,5,digest='different')
    groups,failures,rows=consolidate([a,b,c])
    assert len(groups)==2 and len(failures)==3
    assert all(g['kernel_ms']['median']==3 for g in groups)
