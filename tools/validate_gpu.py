"""Validacion REAL NVIDIA: CPU manual -> PyCUDA directo -> servicio sin recorte."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import threading
import numpy as np
from PIL import Image
import requests
from src.python.filtros_python import crear_filtro_cpu
from webapp.core import GPUBackend, png_bytes
from .client import one
from .common import save_json, environment

TOL=0.001
CASES=[('media',3),('media',9),('media',21),('nitidez',3),('laplaciano',3)]


def error(a,b):
    assert a.shape==b.shape and np.isfinite(b).all()
    dif=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(error_max=float(dif.max()),error_mean=float(dif.mean()),tolerance=TOL)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://localhost:8000');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.url=a.url.rstrip('/');a.out.mkdir(parents=True,exist_ok=False)
    backend=GPUBackend();records=[];passed=False
    save_json(a.out/'environment.json',dict(direct=environment(),gpu=backend.info,
        server=requests.get(a.url+'/api/health',timeout=130).json()))
    try:
        manual=np.arange(1,10,dtype=np.float32).reshape(3,3)
        expected=dict(media=np.array([[12,21,16],[27,45,33],[24,39,28]],dtype=np.float32)/np.float32(9),
            nitidez=np.array([[-1,1,7],[7,5,13],[23,19,31]],dtype=np.float32),
            laplaciano=np.array([[2,1,-4],[-3,0,-7],[-16,-11,-22]],dtype=np.float32))
        y,x=np.indices((19,17))
        arrays=[np.array([[9]],dtype=np.float32),manual,((x*17+y*29)%256).astype(np.float32)]
        for idx,entry in enumerate(arrays):
            path=a.out/f'input_{idx}.pgm';Image.fromarray(entry.astype(np.uint8)).save(path)
            for fil,k in CASES:
                cpu=crear_filtro_cpu(fil,k).ejecutar(entry)
                if idx==1 and k==3:assert error(expected[fil],cpu)['error_max']<=TOL
                for threads in (128,256,512):
                    direct,_=backend.run(entry,fil,k,threads)
                    ecpu=error(cpu,direct);assert ecpu['error_max']<=TOL
                    row,image,raw=one(a.url,path,fil,k,threads,raw=True)
                    assert row['status']=='ok',row
                    service=np.load(BytesIO(raw),allow_pickle=False)
                    eservice=error(direct,service);assert eservice['error_max']<=TOL
                    assert image==png_bytes(direct,fil),'La codificacion visual difiere.'
                    records.append(dict(case=idx,filter=fil,mask=k,threads=threads,
                        request_id=row['request_id'],cpu_direct=ecpu,direct_service=eservice,status='ok'))
                    if idx==2 and k==3 and threads==256:
                        (a.out/f'{fil}_resultado.png').write_bytes(image)
        (a.out/'original.png').write_bytes(png_bytes(arrays[2]))
        # Impulso: signo y magnitudes se validan antes de abs/clip/redondeo.
        impulse=np.zeros((3,3),dtype=np.float32);impulse[1,1]=255
        for fil,center,near in [('nitidez',1275,-255),('laplaciano',-1020,255)]:
            direct,_=backend.run(impulse,fil,3,128)
            assert direct[1,1]==center and direct[0,1]==near
        # Errores reales y recuperacion del mismo servicio.
        invalid=[]
        health=requests.get(a.url+'/api/health',timeout=30).json()
        limit=health['limits']['max_bytes']
        valid=(a.out/'input_1.pgm').read_bytes()
        for name,data,params,expected_status in [
            ('archivo_invalido',b'no soy una imagen',{},422),
            ('archivo_grande',b'x'*(limit+1),{},413),
            ('filtro_desconocido',valid,{'filter':'otro'},400),
            ('mascara_no_permitida',valid,{'filter':'media','mask':7},400)]:
            r=requests.post(a.url+'/api/process',files={'image':('test.png',data)},data=params,timeout=300)
            invalid.append(dict(test=name,http_status=r.status_code,response=r.json()))
            assert r.status_code==expected_status,(name,r.text)
        row,_,_=one(a.url,a.out/'input_1.pgm','media',3,256);assert row['status']=='ok'
        save_json(a.out/'invalid_inputs.json',invalid)
        # Dos imagenes identificables simultaneas; no se comparan relojes de equipos distintos.
        paths=[a.out/'isolation_a.pgm',a.out/'isolation_b.pgm']
        images=[np.full((19,17),v,dtype=np.float32) for v in (23,197)]
        for path,entry in zip(paths,images):Image.fromarray(entry.astype(np.uint8)).save(path)
        barrier=threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(one,a.url,path,'media',3,256,300,True,barrier) for path in paths]
            results=[f.result() for f in futures]
        assert results[0][0]['request_id']!=results[1][0]['request_id']
        isolation=[]
        for entry,(row,_,raw) in zip(images,results):
            assert row['status']=='ok',row
            ref,_=backend.run(entry,'media',3,256)
            e=error(ref,np.load(BytesIO(raw),allow_pickle=False));assert e['error_max']<=TOL
            isolation.append(dict(request_id=row['request_id'],status='ok',**e))
        save_json(a.out/'isolation.json',isolation)
        # Rafaga para observar admision/rechazos; no inventa un rechazo si la GPU termino antes.
        burst=8;barrier=threading.Barrier(burst)
        with ThreadPoolExecutor(max_workers=burst) as pool:
            futures=[pool.submit(one,a.url,a.out/'input_2.pgm','media',21,256,300,False,barrier)
                     for _ in range(burst)]
            burst_rows=[f.result()[0] for f in futures]
        save_json(a.out/'queue_burst.json',burst_rows)
        assert all(r['http_status'] in (200,429) for r in burst_rows)
        passed=True
    finally:
        save_json(a.out/'numerical_validation.json',dict(passed=passed,tolerance=TOL,
            justification='float32, misma acumulacion y --fmad=false. Tolerancia absoluta '
                          'heredada de practica 02; errores calculados en float64 ANTES de visualizar.',
            tests=records,backend='pycuda_real'))
        backend.close()
    print('VALIDACION GPU REAL APROBADA:',len(records),'casos + errores + aislamiento.')

if __name__=='__main__':main()
