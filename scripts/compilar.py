"""Compilar y registrar por separado tiempos de configuracion/build anticipado."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from comun import ahora, guardar_json, sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=Path('build'))
    p.add_argument('--arquitectura',default='86',help='86 para RTX 3070')
    p.add_argument('--cpu-only',action='store_true',help='Solo desarrollo, no sustituye pruebas CUDA')
    a=p.parse_args();build=a.build.resolve();build.mkdir(parents=True,exist_ok=True)
    comandos=[['cmake','-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
        '-DPython3_EXECUTABLE='+sys.executable,'-DCMAKE_CUDA_ARCHITECTURES='+a.arquitectura,
        '-DPRACTICA_ENABLE_CUDA='+('OFF' if a.cpu_only else 'ON')]]
    if platform.system()=='Windows':comandos[0]+=['-G','NMake Makefiles']
    comandos.append(['cmake','--build',str(build),'--config','Release'])
    datos=dict(inicio_utc=ahora(),cpu_only=a.cpu_only,python=sys.executable,etapas=[],
        alcance='Configuracion y compilacion/enlace anticipado completos. Build incremental si ya existia; no kernel_ms.')
    try:
        for nombre,cmd in zip(['configuracion','compilacion_enlace'],comandos):
            inicio=perf_counter()
            result=subprocess.run(cmd)
            datos['etapas'].append(dict(etapa=nombre,comando=cmd,tiempo_ms=(perf_counter()-inicio)*1000,codigo=result.returncode))
            result.check_returncode()
        exe=build/('practica1.exe' if platform.system()=='Windows' else 'practica1')
        if exe.exists():datos['ejecutable_sha256']=sha256(exe)
        datos['estado']='completada'
    finally:
        datos['fin_utc']=ahora();guardar_json(build/'compilacion.json',datos)


if __name__=='__main__':main()
