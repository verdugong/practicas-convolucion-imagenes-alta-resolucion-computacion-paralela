"""Pruebas de regresion: aceleraciones, interrupcion, reanudacion y CSV.
Los datos temporales de estas pruebas NO son resultados de laboratorio.
"""
import argparse
import csv
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from graficar import calcular_aceleraciones, validar_fila, resumir, cobertura
from experimentos import configuraciones, ejecutar, validar_etapa
from comun import errores, leer_filas
from python_filtros import medir
from src.python.filtros_python import FiltroMediaCPU, crear_filtro_cpu
EXE=None


class Pipeline(unittest.TestCase):
    def test_campana_limitacion_resume_y_manipulacion(self):
        # Solo prueba de coordinacion con procesos simulados; no prueba CUDA.
        import experimentos
        from comun import COLUMNAS
        from copy import deepcopy
        plan=[configuraciones(64)[i] for i in (0,7,9)]
        llamadas=[]
        def proceso_simulado(command,log,limite=0):
            def arg(key):return command[command.index(key)+1]
            prefix=Path(arg('--output'));prefix.parent.mkdir(parents=True,exist_ok=True)
            log=Path(log);log.write_text('FIXTURE DE PRUEBA, NO MEDICION GPU.\n')
            es_python=any(str(v).endswith('/python_filtros.py') or str(v).endswith('\\python_filtros.py') for v in command)
            es_pycuda=any(str(v).endswith('/pycuda_filtros.py') or str(v).endswith('\\pycuda_filtros.py') for v in command)
            impls=['PYTHON'] if es_python else ['PYCUDA'] if es_pycuda else ['CPU','CUDA']
            suffix='_python_raw.csv' if es_python else '_pycuda_raw.csv' if es_pycuda else '_raw.csv'
            with Image.open(arg('--input')) as im:n=im.width
            t=int(arg('--threads')) if '--threads' in command else 0
            timeout=es_python and n==4000 and limite>0
            with Path(str(prefix)+suffix).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=COLUMNAS);w.writeheader()
                for impl in impls:
                    for r in range(1,2 if timeout else 6):
                        gpu=impl in ('CUDA','PYCUDA')
                        w.writerow(dict(implementacion=impl,filtro=arg('--filter'),mascara=int(arg('--mask')),
                            ancho=n,alto=n,hilos=t if gpu else 0,bloques=(n*n+t-1)//t if gpu else 0,
                            repeticion=r,total_ms=10,kernel_ms=2 if gpu else '',h2d_ms=1 if gpu else '',
                            d2h_ms=1 if gpu else '',error_max=0,error_medio=0,tolerancia=.001,
                            validado='true',validado_cuda='true',error_cuda_max=0,error_cuda_medio=0,
                            unidad='ms',estado='completada'))
            llamadas.append(impls)
            return dict(codigo=130 if timeout else 0,motivo='Presupuesto agotado' if timeout else '',
                        limite_s=limite,tiempo_proceso_s=1)
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            td=Path(td);data=td/'data';data.mkdir();exe=td/'fixture.exe';exe.write_bytes(b'fixture')
            for n in (2000,4000,6000):Image.new('L',(n,n),1).save(data/f'imagen_{n}.pgm')
            out=td/'salida'
            base=['experimentos','--exe',str(exe),'--data',str(data),'--alternativa-lado','64']
            def check(command,**kwargs):return subprocess.CompletedProcess(command,0,'{}','')
            with patch.object(experimentos,'configuraciones',lambda lado:deepcopy(plan)),\
                 patch.object(experimentos,'capturar',lambda cmd:dict(stdout='fixture')),\
                 patch.object(experimentos.subprocess,'run',check),\
                 patch.object(experimentos,'ejecutar',proceso_simulado),\
                 patch('graficar.graficas'),patch('graficar.comparativos'):
                with patch.object(sys,'argv',base+['--out',str(out),'--python-max-segundos','.1']):experimentos.main()
                cov=json.loads((out/'cobertura.json').read_text())
                self.assertTrue(cov['cobertura_experimental_satisfecha'])
                self.assertEqual(cov['configuraciones_python_con_limitacion_documentada'],['media_4000_k3_t256'])
                antes=len(llamadas)
                with patch.object(sys,'argv',base+['--resume',str(out)]):experimentos.main()
                self.assertEqual(antes,len(llamadas))
                with patch.object(sys,'argv',base+['--resume',str(out),'--reintentar-python','--python-max-segundos','0']):experimentos.main()
                self.assertEqual(antes+1,len(llamadas))
                cov=json.loads((out/'cobertura.json').read_text())
                self.assertEqual(cov['configuraciones_python_con_limitacion_documentada'],[])
                m=json.loads((out/'experimentos.json').read_text())
                prefix=out/m['configuraciones'][0]['etapas']['PYTHON']['prefijo']
                with Path(str(prefix)+'_python_raw.csv').open('a') as f:f.write('\n')
                with patch.object(sys,'argv',base+['--resume',str(out)]),contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):experimentos.main()

    def test_bases_aceleracion(self):
        r=calcular_aceleraciones({'CPU':{'total_ms':100},'PYTHON':{'total_ms':10000},
            'CUDA':{'kernel_ms':2,'total_ms':10},'PYCUDA':{'kernel_ms':4,'total_ms':20}})
        self.assertEqual(r['CUDA']['aceleracion_kernel'],50)
        self.assertEqual(r['PYCUDA']['aceleracion_kernel'],2500)
        self.assertEqual(r['PYCUDA']['aceleracion_total'],500)
        self.assertEqual(r['PYCUDA']['referencia_aceleracion'],'PYTHON')
        self.assertEqual(r['comparacion_gpu']['razon_cuda_pycuda_total'],.5)

    def test_incompleto_no_aceleracion(self):
        r=calcular_aceleraciones({'CPU':{'total_ms':100},'PYCUDA':{'kernel_ms':4,'total_ms':20}})
        self.assertNotIn('PYCUDA',r)

    def test_plan_no_repite_python_hilos(self):
        plan=configuraciones()
        self.assertEqual(len(plan),14)
        self.assertEqual(sum(not c['alternativa'] for c in plan),9)
        self.assertEqual(sum('PYTHON' in c['requeridas'] and not c['alternativa'] for c in plan),7)
        self.assertEqual(len({c['prefijo'] for c in plan}),14)

    def test_nan_no_valida(self):
        a=np.array([[1]],dtype=np.float32)
        b=np.array([[np.nan]],dtype=np.float32)
        self.assertTrue(np.isinf(errores(a,b)[0]))

    def test_interrupcion_reanudacion(self):
        a=np.arange(9,dtype=np.float32).reshape(3,3)
        ref=FiltroMediaCPU(3).ejecutar(a)
        original=FiltroMediaCPU.aplicar
        cuenta=[0]
        def aplicar(obj,entrada,salida):
            cuenta[0]+=1
            if cuenta[0]==4:raise KeyboardInterrupt
            return original(obj,entrada,salida)
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            pref=Path(td)/'caso'
            with patch.object(FiltroMediaCPU,'aplicar',aplicar):
                with self.assertRaises(KeyboardInterrupt):medir(a,ref,'media',3,5,pref)
            rows=leer_filas(str(pref)+'_python_raw.csv')
            self.assertEqual(len(rows),1)
            primera=rows[0].copy()
            estado=json.loads(Path(str(pref)+'_python_estado.json').read_text())
            self.assertEqual(estado['estado'],'no_completada')
            self.assertEqual(estado['fase'],'convolucion')
            medir(a,ref,'media',3,5,pref,True)
            rows=leer_filas(str(pref)+'_python_raw.csv')
            self.assertEqual(len(rows),5)
            self.assertEqual(rows[0],primera)
            self.assertEqual([int(r['repeticion']) for r in rows],[1,2,3,4,5])
            c=dict(filtro='media',mascara=3,resolucion=3,hilos=256)
            self.assertTrue(validar_etapa(pref,'PYTHON',c,5))
            self.assertTrue(all(validar_fila(r,c,'PYTHON') for r in rows))
            mal=rows[0].copy();mal['total_ms']='nan'
            self.assertFalse(validar_fila(mal,c,'PYTHON'))

    def test_interrupcion_calentamiento(self):
        a=np.ones((3,3),dtype=np.float32);ref=FiltroMediaCPU(3).ejecutar(a)
        original=FiltroMediaCPU.aplicar;cuenta=[0]
        def aplicar(obj,entrada,salida):
            cuenta[0]+=1
            if cuenta[0]==2:raise KeyboardInterrupt
            return original(obj,entrada,salida)
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            pref=Path(td)/'caso'
            with patch.object(FiltroMediaCPU,'aplicar',aplicar):
                with self.assertRaises(KeyboardInterrupt):medir(a,ref,'media',3,5,pref)
            estado=json.loads(Path(str(pref)+'_python_estado.json').read_text())
            self.assertEqual(estado['fase'],'calentamiento')
            self.assertEqual(estado['repeticiones_completas'],0)
            self.assertFalse(Path(str(pref)+'_python_raw.csv').exists())

    def test_presupuesto_proceso(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            td=Path(td)
            script=td/'espera.py'
            script.write_text('import time\ntry:\n while True: time.sleep(.01)\nexcept KeyboardInterrupt:\n pass\n')
            r=ejecutar([sys.executable,str(script)],td/'log.txt',.5)
            self.assertTrue(r['motivo'])
            self.assertGreaterEqual(r['tiempo_proceso_s'],.5)

    def test_datos_incompletos_conservados_sin_estadistica(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            td=Path(td);a=np.ones((3,3),dtype=np.float32)
            ref=FiltroMediaCPU(3).ejecutar(a);pref=td/'medido'
            medir(a,ref,'media',3,5,pref)
            c=dict(filtro='media',resolucion=3,mascara=3,hilos=256,prefijo='caso',
                   experimentos=['test'],requeridas=['PYTHON'],etapas={
                       'PYTHON':dict(prefijo='medido',estado='completada')})
            m=dict(formato=2,repeticiones=5,configuraciones=[c])
            (td/'experimentos.json').write_text(json.dumps(m))
            _,res,raw,states=resumir(td)
            self.assertEqual(len(res),1)
            self.assertEqual(res[0]['aceleracion_total'],'')
            csvpath=Path(str(pref)+'_python_raw.csv')
            lines=csvpath.read_text().splitlines();csvpath.write_text('\n'.join(lines[:2])+'\n')
            _,res,raw,states=resumir(td)
            self.assertEqual(len(res),0)
            self.assertEqual(len(raw),1)
            self.assertEqual(next(s for s in states if s['implementacion']=='PYTHON')['estado'],'no_completada')

    def test_referencia_cpp_y_python_rectangulares(self):
        if EXE is None:self.skipTest('Usar --exe para validacion C++/Python.')
        rng=np.random.default_rng(123)
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            for h,w in [(1,1),(3,3),(19,17),(7,23)]:
                a=rng.integers(0,256,(h,w),dtype=np.uint8)
                ruta=td/f'{h}_{w}.pgm';Image.fromarray(a).save(ruta)
                for filtro,k in [('media',3),('media',9),('media',21),('nitidez',3),('laplaciano',3)]:
                    pref=td/f'{h}_{w}_{filtro}_{k}'
                    subprocess.run([str(EXE),'--input',str(ruta),'--filter',filtro,'--mask',str(k),
                        '--reps','5','--output',str(pref),'--export-float'],check=True,capture_output=True)
                    c=np.fromfile(str(pref)+'_cpu.f32',dtype='<f4').reshape(h,w)
                    py=crear_filtro_cpu(filtro,k).ejecutar(a.astype(np.float32))
                    self.assertLessEqual(errores(c,py)[0],.001)
                    rows=leer_filas(str(pref)+'_raw.csv')
                    self.assertTrue(all(r['estado']=='completada' for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path)
    a,rest=p.parse_known_args()
    EXE=a.exe.resolve() if a.exe else None
    unittest.main(argv=[sys.argv[0],*rest])
