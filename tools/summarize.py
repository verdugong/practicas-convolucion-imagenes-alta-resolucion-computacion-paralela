"""Consolidar SOLO mediciones reales completas. No convertir fallos en tiempo cero."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
from .common import save_json

METRICS=['h2d_ms','kernel_ms','d2h_ms','total_ms','allocation_ms','decode_prepare_ms',
         'encode_ms','server_processing_ms','queue_ms','client_total_ms','processing_ms']
KEYS=['mode','experiment','filter','width','height','mask','threads','concurrency','environment','sha256_input']


def consolidate(paths):
    groups=defaultdict(list);failures=[];all_rows=[]
    for path in paths:
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                row['_source']=str(path);all_rows.append(row)
                if row.get('status')!='ok':failures.append(row);continue
                if row.get('phase')!='stable':continue
                groups[tuple(row.get(k,'') for k in KEYS)].append(row)
    summaries=[]
    for key,rows in groups.items():
        complete=len(rows)>=5
        summary=dict(zip(KEYS,key),n=len(rows),complete=complete,
                     input_hashes=sorted(set(r['sha256_input'] for r in rows)))
        for metric in METRICS:
            vals=[float(r[metric]) for r in rows if r.get(metric)]
            if len(vals)>=5:
                q1,q3=np.quantile(vals,[.25,.75]);summary[metric]=dict(
                    median=float(np.median(vals)),iqr=float(q3-q1),n=len(vals))
        if summary['experiment']=='concurrencia':
            rounds={r['round']:float(r['successful_per_second']) for r in rows}
            if len(rounds)>=5:
                vals=list(rounds.values());q1,q3=np.quantile(vals,[.25,.75])
                summary['successful_per_second']=dict(median=float(np.median(vals)),
                    iqr=float(q3-q1),rounds=len(rounds))
        summaries.append(summary)
    return summaries,failures,all_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,default=Path('evidence'))
    p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    paths=sorted(a.evidence.rglob('direct.csv'))+sorted(a.evidence.rglob('service.csv'))
    if not paths:raise SystemExit('No hay CSV medidos. Ejecute direct/client antes de resumir.')
    summaries,failures,rows=consolidate(paths)
    save_json(a.out/'summary.json',dict(groups=summaries,failures=failures,
        csv_files=[str(p) for p in paths],warnings=[
            'No atribuir diferencias a Docker sin host equivalente, misma GPU, hash e hilos.',
            'No usar mediciones simuladas ni resultados de practica 02 como resultados de practica 03.']))
    with (a.out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        fields=KEYS+['n','complete']+[m+'_'+suffix for m in METRICS for suffix in ('median','iqr')]
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for s in summaries:
            record=s.copy()
            for m in METRICS:
                for suffix in ('median','iqr'):record[m+'_'+suffix]=s.get(m,{}).get(suffix,'')
            w.writerow(record)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ['kernel_ms','server_processing_ms','client_total_ms','queue_ms','successful_per_second']:
        valid=[s for s in summaries if metric in s and s['complete']]
        if not valid:continue
        labels=[f"{s['mode']} / {s['filter']} k{s['mask']}\n{s['width']}x{s['height']} / c{s['concurrency']} / {s['experiment']}" for s in valid]
        fig,ax=plt.subplots(figsize=(12,max(4,len(valid)*.4)))
        ax.barh(range(len(valid)),[s[metric]['median'] for s in valid],color='#275c50')
        ax.set_yticks(range(len(valid)),labels,fontsize=8);ax.invert_yaxis()
        ax.set_xlabel('solicitudes exitosas / segundo' if metric=='successful_per_second' else 'Mediana (ms)')
        ax.set_title(metric+' · solo series con al menos cinco observaciones')
        fig.tight_layout();fig.savefig(a.out/(metric+'.png'),dpi=160);plt.close(fig)
    print('Resumen:',a.out,'; grupos:',len(summaries),'; fallos:',len(failures))

if __name__=='__main__':main()
