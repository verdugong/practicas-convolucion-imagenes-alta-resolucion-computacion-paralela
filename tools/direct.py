"""PyCUDA directo sin HTTP. Mismos kernels y preparacion usados por el servicio."""
import argparse
from pathlib import Path
import time
import uuid
from webapp.core import GPUBackend, Limits, process
from .common import environment, save_json, writer, sha


def plan(data, small=False):
    # Los experimentos comparten media 9/4000 cuando coinciden las condiciones.
    sizes=(512,) if small else (2000,4000,6000)
    cases=[(data/f'imagen_{s}.pgm','media',9,'resolucion') for s in sizes]
    base=data/('imagen_512.pgm' if small else 'imagen_4000.pgm')
    cases += [(base,f,k,'filtros_mascara') for f,k in
              [('media',3),('media',21),('nitidez',3),('laplaciano',3)]]
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['host_direct','container_direct'],required=True)
    p.add_argument('--data',type=Path,default=Path('data/preparadas'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,choices=[128,256,512],default=256)
    p.add_argument('--reps',type=int,default=5)
    p.add_argument('--small',action='store_true',help='Alternativa 512: documentar limitacion; no reemplaza pruebas viables')
    a=p.parse_args()
    if a.reps<5:p.error('Se requieren al menos cinco repeticiones.')
    a.out.mkdir(parents=True,exist_ok=False)
    env=environment();save_json(a.out/'environment.json',env)
    f,w=writer(a.out/'direct.csv');backend=None
    first_success=True
    try:
        backend=GPUBackend();save_json(a.out/'gpu_init.json',backend.info)
        for path,fil,k,exp in plan(a.data,a.small):
            if not path.is_file():raise FileNotFoundError(path)
            data=path.read_bytes()
            for rep in range(-1,a.reps+1):
                phase='cold_first' if rep==-1 else 'warmup' if rep==0 else 'stable'
                row=dict(request_id=uuid.uuid4().hex,mode=a.mode,experiment=exp,filter=fil,
                    mask=k,threads=a.threads,repetition=max(rep,0),concurrency=1,phase=phase,
                    environment=env['hostname'],unit='ms',sha256_input=sha(path))
                start=time.perf_counter()
                try:
                    result=process(data,fil,k,a.threads,backend,Limits())
                    row.update(result['metrics'],status='ok')
                    if first_success:
                        save_json(a.out/'cold_start.json',dict(context_init_ms=backend.info['context_init_ms'],
                            compile_load_ms=backend.info['compile_load_ms'],first_processing_ms=row['processing_ms'],
                            scope='Carga contexto + compilacion/carga y primera ejecucion registradas por separado; '
                                  'no incluye importaciones previas, arranque Python ni Docker. Cache puede existir.'))
                        first_success=False
                    # La escritura a disco de evidencia queda FUERA de processing_ms directo.
                    if rep==a.reps:
                        (a.out/f'{fil}_{k}_{path.stem}.png').write_bytes(result['png'])
                except Exception as exc:
                    row.update(status='no_completada',error=str(exc),client_total_ms=(time.perf_counter()-start)*1000)
                w.writerow(row);f.flush()
                print(a.mode,path.name,fil,k,phase,row['status'],row.get('kernel_ms'),flush=True)
    finally:
        if backend:backend.close()
        f.close()

if __name__=='__main__':main()
