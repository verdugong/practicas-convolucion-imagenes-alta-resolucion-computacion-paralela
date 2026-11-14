"""Compilar un ejecutable solo CUDA C sin cambiar CMake ni la campana original."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def comando(arquitectura):
    if not arquitectura.isdigit():raise ValueError('Arquitectura numerica, por ejemplo 86 para RTX 3070.')
    nvcc=shutil.which('nvcc')
    if not nvcc:raise RuntimeError('No se encuentra nvcc. Abrir terminal de Visual Studio con CUDA Toolkit.')
    out=HERE/'bin'/('cuda_individual.exe' if sys.platform=='win32' else 'cuda_individual')
    sources=[HERE/'cuda_individual.cpp',ROOT/'src/common/imagen.cpp',
        ROOT/'src/gpu/filtros_gpu.cu',ROOT/'src/gpu/media_gpu.cu',
        ROOT/'src/gpu/nitidez_gpu.cu',ROOT/'src/gpu/laplaciano_gpu.cu']
    if any(not p.is_file() for p in sources):
        raise FileNotFoundError('Colocar defensa_gpu junto a src e include, dentro de Practica2.')
    cmd=[nvcc,'-std=c++17','-O3','--fmad=false','-arch=sm_'+arquitectura,
         '-I'+str(ROOT/'include'),*[str(p) for p in sources],'-o',str(out)]
    if sys.platform=='win32':cmd+=['-Xcompiler=/EHsc']
    return cmd,out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arquitectura',default='86',help='86 para RTX 3070; cambiar para otra GPU')
    a=p.parse_args();cmd,out=comando(a.arquitectura);out.parent.mkdir(exist_ok=True)
    print(subprocess.list2cmdline(cmd),flush=True)
    inicio=perf_counter();subprocess.run(cmd,check=True,cwd=ROOT)
    (HERE/'bin/compilacion.json').write_text(json.dumps(dict(comando=cmd,
        compilacion_ms=(perf_counter()-inicio)*1000,arquitectura=a.arquitectura),indent=2))
    print('Ejecutable listo:',out)


if __name__=='__main__':
    try:main()
    except (OSError,RuntimeError,ValueError,subprocess.CalledProcessError) as exc:
        print('ERROR:',exc,file=sys.stderr);sys.exit(1)
