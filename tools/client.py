"""Cliente HTTP: el tiempo incluye POST y descarga completa del PNG final."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from pathlib import Path
import threading
import time
import requests
from PIL import Image
from .common import environment, save_json, writer, sha
from .direct import plan


def one(url,path,fil,k,threads,timeout=300,raw=False,barrier=None):
    # Lectura local fuera del cronometro del cliente, igual para todos los casos.
    content=Path(path).read_bytes()
    digest=sha(path)
    with Image.open(path) as im: width,height=im.size
    if barrier:barrier.wait(timeout=30)
    start=time.perf_counter()
    row=dict(status='error',unit='ms',sha256_input=digest,filter=fil,mask=k,threads=threads,width=width,height=height)
    try:
        r=requests.post(url+'/api/process',files={'image':(Path(path).name,content)},
            data=dict(filter=fil,mask=k,threads=threads,raw=int(raw)),timeout=timeout)
        row['http_status']=r.status_code
        body=r.json();row['request_id']=body.get('request_id')
        if not r.ok:
            row['error']=body.get('message',r.text)
            return row,None,None
        row.update(body['metrics'])
        image=requests.get(url+body['result_url'],timeout=timeout);image.raise_for_status()
        image_bytes=image.content
        row['client_total_ms']=(time.perf_counter()-start)*1000
        raw_bytes=None
        if raw:
            rr=requests.get(url+body['raw_url'],timeout=timeout);rr.raise_for_status();raw_bytes=rr.content
        row['status']='ok'
        return row,image_bytes,raw_bytes
    except Exception as exc:
        row['error']=str(exc)
        return row,None,None
    finally:
        row.setdefault('client_total_ms',(time.perf_counter()-start)*1000)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://localhost:8000')
    p.add_argument('--data',type=Path,default=Path('data/preparadas'))
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--threads',type=int,choices=[128,256,512],default=256)
    p.add_argument('--reps',type=int,default=5)
    p.add_argument('--network-note',required=True,help='Ej. mismo host/loopback o laptop Wi-Fi LAN')
    p.add_argument('--small',action='store_true')
    a=p.parse_args();a.url=a.url.rstrip('/')
    if a.reps<5:p.error('Minimo cinco repeticiones/rondas.')
    a.out.mkdir(parents=True,exist_ok=False)
    health=requests.get(a.url+'/api/health',timeout=130);health.raise_for_status()
    save_json(a.out/'environment.json',dict(client=environment(),server=health.json(),
        url=a.url,network_note=a.network_note,small_alternative=a.small))
    f,w=writer(a.out/'service.csv')
    try:
        for path,fil,k,exp in plan(a.data,a.small):
            for rep in range(-1,a.reps+1):
                row,image,_=one(a.url,path,fil,k,a.threads)
                row.update(mode='service',experiment=exp,repetition=max(rep,0),concurrency=1,
                    phase='cold_first' if rep==-1 else 'warmup' if rep==0 else 'stable',
                    environment=a.network_note)
                w.writerow(row);f.flush()
                if image and rep==a.reps:(a.out/f'{fil}_{k}_{path.stem}.png').write_bytes(image)
                print(path.name,fil,k,row['phase'],row['status'],flush=True)
        path=a.data/('imagen_512.pgm' if a.small else 'imagen_4000.pgm')
        for concurrency in (1,2):
            # Calentamiento de la misma configuracion antes de las rondas.
            one(a.url,path,'media',9,a.threads)
            for rnd in range(1,a.reps+1):
                barrier=threading.Barrier(concurrency)
                start=time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures=[pool.submit(one,a.url,path,'media',9,a.threads,300,False,barrier)
                             for _ in range(concurrency)]
                    rows=[r.result()[0] for r in futures]
                elapsed=time.perf_counter()-start
                rate=sum(r['status']=='ok' for r in rows)/elapsed
                for i,row in enumerate(rows,1):
                    row.update(mode='service',experiment='concurrencia',repetition=i,round=rnd,
                        concurrency=concurrency,phase='stable',round_ms=elapsed*1000,
                        successful_per_second=rate,environment=a.network_note)
                    w.writerow(row)
                f.flush();print('Ronda',concurrency,rnd,'exitos/s',rate,flush=True)
    finally:f.close()

if __name__=='__main__':main()
