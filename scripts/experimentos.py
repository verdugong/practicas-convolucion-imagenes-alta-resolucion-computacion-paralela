"""Campana reproducible: GPU primero, alternativa comun y Python con presupuesto."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
from time import perf_counter
import numpy as np
import psutil
from PIL import Image, __version__ as PILLOW_VERSION
from comun import ahora, guardar_json, leer_filas, sha256

ROOT = Path(__file__).resolve().parents[1]


def configuraciones(lado=512):
    base = [("media",4000,3,256,["filtros","mascaras"]),
            ("nitidez",4000,3,256,["filtros"]),
            ("laplaciano",4000,3,256,["filtros"]),
            ("media",2000,9,256,["resoluciones"]),
            ("media",4000,9,256,["resoluciones","mascaras","hilos"]),
            ("media",6000,9,256,["resoluciones"]),
            ("media",4000,21,256,["mascaras"]),
            ("media",4000,9,128,["hilos"]),
            ("media",4000,9,512,["hilos"])]
    base += [(f,lado,k,256,["alternativa"]) for f,k in
             [("media",3),("nitidez",3),("laplaciano",3),("media",9),("media",21)]]
    return [dict(filtro=f,resolucion=n,mascara=k,hilos=t,experimentos=g,
                 prefijo=f'{f}_{n}_k{k}_t{t}', alternativa=g==["alternativa"],
                 requeridas=["CUDA","PYCUDA"] if g==["hilos"] else
                 ["CPU","CUDA","PYTHON","PYCUDA"], etapas={}) for f,n,k,t,g in base]


def capturar(command):
    try:
        r = subprocess.run(command,capture_output=True,text=True,errors="replace",timeout=30)
        return dict(command=command,codigo=r.returncode,stdout=r.stdout,stderr=r.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:
        return dict(command=command,error=str(e))


def huella_codigo():
    archivos = [ROOT/'CMakeLists.txt', ROOT/'requirements.txt']
    for folder in ('src','include','scripts','tests'):
        archivos += [p for p in (ROOT/folder).rglob('*') if p.is_file()
                     and p.suffix in ('.py','.cpp','.cu','.cuh','.hpp')]
    registro = {str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in sorted(archivos)}
    return hashlib.sha256(json.dumps(registro,sort_keys=True).encode()).hexdigest(), registro


def detener(proceso):
    if proceso.poll() is not None:
        return
    try:
        if os.name == 'nt':
            proceso.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(proceso.pid, signal.SIGINT)
        proceso.wait(timeout=10)
    except (OSError,subprocess.TimeoutExpired):
        proceso.kill()
        proceso.wait()


def ejecutar(command, log, limite=0):
    """Un proceso a la vez. Limite incluye arranque/preparacion, no es kernel_ms."""
    kwargs = {'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
    log = Path(log)
    log.parent.mkdir(parents=True,exist_ok=True)
    inicio=perf_counter()
    with log.open('w',encoding='utf-8') as out:
        proceso=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,errors='replace',bufsize=1,**kwargs)
        def consumir():
            for linea in proceso.stdout:
                print(linea,end='',flush=True)
                out.write(linea)
                out.flush()
        hilo=threading.Thread(target=consumir,daemon=True)
        hilo.start()
        motivo=''
        try:
            proceso.wait(timeout=limite if limite else None)
        except subprocess.TimeoutExpired:
            motivo=f'Presupuesto de laboratorio agotado: {limite:g} s por proceso Python.'
            detener(proceso)
        except KeyboardInterrupt:
            detener(proceso)
            raise
        finally:
            hilo.join(timeout=15)
            proceso.stdout.close()
        return dict(codigo=proceso.returncode,tiempo_proceso_s=perf_counter()-inicio,
                    motivo=motivo,limite_s=limite)


def validar_etapa(prefix, etapa, c, reps):
    from graficar import validar_fila
    suffix={'CPU_CUDA':'_raw.csv','PYCUDA':'_pycuda_raw.csv','PYTHON':'_python_raw.csv'}[etapa]
    rows=leer_filas(str(prefix)+suffix)
    impls=['CPU','CUDA'] if etapa=='CPU_CUDA' else [etapa]
    for impl in impls:
        subset=[r for r in rows if r.get('implementacion')==impl]
        try:
            if len(subset)!=reps or {int(r['repeticion']) for r in subset}!=set(range(1,reps+1)):
                return False
        except (KeyError,ValueError):return False
        if not all(validar_fila(r,c,impl) for r in subset):return False
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,default=Path('build/practica1.exe' if os.name=='nt' else 'build/practica1'))
    p.add_argument('--data',type=Path,default=Path('data/preparadas'))
    p.add_argument('--out',type=Path)
    p.add_argument('--resume',type=Path,help='Carpeta de una campana v2, mismo equipo/codigo/entradas')
    p.add_argument('--reps',type=int,default=5)
    p.add_argument('--python-max-segundos',type=float,default=120,
                   help='Presupuesto por proceso Python grande (incluye calentamiento); 0 sin limite')
    p.add_argument('--alternativa-lado',type=int,default=512,help='Resolucion comun adicional (64..1000)')
    p.add_argument('--motivo-interrupcion',default='Interrupcion manual del usuario (Ctrl+C).')
    p.add_argument('--reintentar-python',action='store_true',help='Reintentar casos que agotaron presupuesto')
    p.add_argument('--x',type=int,default=1600)
    p.add_argument('--y',type=int,default=600)
    p.add_argument('--lado',type=int,default=600)
    a=p.parse_args()
    if not 5<=a.reps<=1000 or not 64<=a.alternativa_lado<=1000 or a.python_max_segundos<0:
        p.error('Parametros fuera de rango.')
    if a.resume and a.out:
        p.error('Elegir --resume o --out.')
    if not a.exe.is_file(): p.error('Compilar primero el ejecutable CUDA.')
    if a.x<0 or a.y<0 or a.lado<=0 or max(a.x,a.y)+a.lado>4000:
        p.error('Recorte fuera de la imagen de 4000.')
    a.exe=a.exe.resolve(); a.data=a.data.resolve()
    entradas={}
    for n in (2000,4000,6000):
        path=a.data/f'imagen_{n}.pgm'
        with path.open('rb') as f: magic=f.read(2)
        with Image.open(path) as im:
            if im.size!=(n,n) or im.mode!='L' or magic!=b'P5': p.error(f'Entrada invalida: {path}')
        entradas[str(n)]={'sha256':sha256(path),'path':str(path)}
    check=subprocess.run([str(a.exe),'--cuda-disponible'],capture_output=True)
    if check.returncode: p.error('El ejecutable no tiene CUDA habilitado.')
    check=subprocess.run([sys.executable,str(ROOT/'scripts/pycuda_filtros.py'),'--check'],
                         capture_output=True,text=True,errors='replace')
    if check.returncode: p.error('PyCUDA no esta listo:\n'+check.stdout+check.stderr)
    gpu=json.loads(check.stdout.strip().splitlines()[-1])
    codigo,archivos=huella_codigo()
    version_path=ROOT/'VERSION_CODIGO.json'
    version=json.loads(version_path.read_text(encoding='utf-8')) if version_path.exists() else {}
    version_codigo=dict(commit_base_distribucion=version.get('commit','no_identificado'),
        codigo_coincide_con_distribucion=version.get('codigo_sha256')==codigo)
    entorno=dict(sistema=platform.platform(),maquina=platform.node(),cpu=platform.processor(),
        fisicos=psutil.cpu_count(logical=False),logicos=psutil.cpu_count(),ram=psutil.virtual_memory().total,
        python=sys.version,numpy=np.__version__,pillow=PILLOW_VERSION,gpu=gpu,
        nvcc=capturar(['nvcc','--version']).get('stdout',''),
        gpu_identidad=capturar(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader']).get('stdout',''))
    firma=dict(codigo=codigo,exe=sha256(a.exe),entorno=entorno,
               entradas={k:v['sha256'] for k,v in entradas.items()})
    output=(a.resume or a.out or Path('results')/datetime.now().strftime('campana_%Y%m%d_%H%M%S')).resolve()
    manifest_path=output/'experimentos.json'
    if a.resume:
        m=json.loads(manifest_path.read_text(encoding='utf-8'))
        if m.get('formato')!=2: p.error('No se reanuda una campana antigua con codigo nuevo. Ver evidencia_original.')
        if m['firma']!=firma: p.error('Cambio equipo/software/codigo/ejecutable/entradas. Iniciar nueva campana.')
        a.reps=m['repeticiones'];a.alternativa_lado=m['alternativa_lado']
        for c in m['configuraciones']:
            for key,s in c['etapas'].items():
                if s['estado']=='completada' and not validar_etapa(output/s['prefijo'],key,c,a.reps):
                    p.error('Resultados completos alterados/incompletos: '+c['prefijo']+' '+key)
                for relativo,esperado in s.get('archivos_sha256',{}).items():
                    if not (output/relativo).is_file() or sha256(output/relativo)!=esperado:
                        p.error('Archivo de resultados modificado: '+relativo)
    else:
        if output.exists():p.error('La carpeta ya existe: usar --resume o un nombre nuevo.')
        output.mkdir(parents=True)
        alternativa=output/'entrada_alternativa.pgm'
        with Image.open(a.data/'imagen_4000.pgm') as im:
            im.resize((a.alternativa_lado,a.alternativa_lado),Image.Resampling.LANCZOS).save(alternativa)
        m=dict(formato=2,inicio_utc=ahora(),estado='en_curso',repeticiones=a.reps,
            alternativa_lado=a.alternativa_lado,entradas=entradas,firma=firma,archivos_codigo=archivos,
            version_codigo=version_codigo,
            alternativa=dict(path=alternativa.name,sha256=sha256(alternativa),
                origen='imagen_4000.pgm',metodo='Pillow LANCZOS, gris; entrada adicional, no sustituye originales'),
            configuraciones=configuraciones(a.alternativa_lado),sesiones=[],
            recorte=dict(x=a.x,y=a.y,lado=a.lado),
            alcance='CPU solo convolucion; GPU H2D/kernel/D2H/total; reservas excluidas',
            herramientas=[capturar(cmd) for cmd in [['nvcc','--version'],['nvidia-smi'],
                ['git','rev-parse','HEAD'],['git','status','--short'],[sys.executable,'-m','pip','freeze']]])
    if sha256(output/m['alternativa']['path'])!=m['alternativa']['sha256']:
        p.error('La entrada alternativa fue modificada.')
    sesion=dict(inicio_utc=ahora(),python_max_segundos=a.python_max_segundos,estado='en_curso')
    m['sesiones'].append(sesion);m['estado']='en_curso'
    def save(): guardar_json(manifest_path,m)
    save()
    try:
        # Primero toda la campana GPU, incluyendo la alternativa.
        trabajos=[(c,stage) for c in m['configuraciones'] for stage in ('CPU_CUDA','PYCUDA')]
        # Alternativa completa primero: cinco filtros/mascaras sobre la misma entrada pequena.
        trabajos += [(c,'PYTHON') for c in sorted(m['configuraciones'],key=lambda c:not c['alternativa'])
                     if 'PYTHON' in c['requeridas']]
        for c,etapa in trabajos:
            anterior=c['etapas'].get(etapa,{})
            if anterior.get('estado')=='completada': continue
            if (etapa=='PYTHON' and anterior.get('estado')=='no_completada'
                and anterior.get('limite_s',0)>0 and not a.reintentar_python): continue
            if anterior.get('estado')=='fallida':
                raise RuntimeError('Etapa fallida requiere diagnostico y nueva campana: '+c['prefijo']+' '+etapa)
            # Python reanuda CSV del mismo codigo/entorno; CUDA reinicia una etapa interrumpida en otro prefijo.
            historial=anterior.get('intentos',[])
            intento=len(historial)+1
            rel=(anterior.get('prefijo') if etapa=='PYTHON' else None) or f"mediciones/{c['prefijo']}/{etapa.lower()}_{intento}"
            prefix=output/rel;prefix.parent.mkdir(parents=True,exist_ok=True)
            estado=dict(estado='en_curso',prefijo=rel,intentos=historial)
            c['etapas'][etapa]=estado
            entrada=output/m['alternativa']['path'] if c['alternativa'] else a.data/f"imagen_{c['resolucion']}.pgm"
            comun=['--input',str(entrada),'--filter',c['filtro'],'--mask',str(c['mascara']),
                   '--reps',str(a.reps),'--output',str(prefix)]
            if etapa=='CPU_CUDA':
                command=[str(a.exe),*comun,'--threads',str(c['hilos']),'--export-float']
            else:
                referencia=output/c['etapas']['CPU_CUDA']['prefijo']
                script='python_filtros.py' if etapa=='PYTHON' else 'pycuda_filtros.py'
                command=[sys.executable,str(ROOT/'scripts'/script),*comun,'--reference-prefix',str(referencia)]
                if etapa=='PYCUDA':command+=['--threads',str(c['hilos'])]
                elif anterior:command+=['--resume']
            limite=a.python_max_segundos if etapa=='PYTHON' and not c['alternativa'] else 0
            registro=dict(inicio_utc=ahora(),limite_s=limite,comando=command,
                          log=f'{rel}_sesion{intento}.log',estado='en_curso')
            historial.append(registro);save()
            print(f"\n{c['prefijo']} / {etapa} / intento {intento}",flush=True)
            t0=perf_counter()
            try:
                result=ejecutar(command,output/registro['log'],limite)
                registro.update(result)
                if result['codigo']==0 and validar_etapa(prefix,etapa,c,a.reps):
                    estado['estado']='completada'
                    estado['archivos_sha256']={str(f.relative_to(output)):sha256(f)
                        for f in prefix.parent.glob(prefix.name+'*')
                        if f.is_file() and f.suffix in ('.csv','.pgm','.f32')}
                elif result['motivo']:
                    estado['estado']='no_completada'
                else:
                    estado['estado']='fallida'
                estado['motivo']=result['motivo'] or ('' if estado['estado']=='completada' else 'Proceso o validacion fallidos; revisar log.')
                estado['limite_s']=limite
            except KeyboardInterrupt:
                estado['estado']='no_completada';estado['limite_s']=0
                estado['motivo']=a.motivo_interrupcion
                registro['tiempo_proceso_s']=perf_counter()-t0
                raise
            except Exception as exc:
                estado['estado']='fallida';estado['motivo']=str(exc)
                registro['tiempo_proceso_s']=perf_counter()-t0
                raise
            finally:
                registro['estado']=estado['estado'];registro['fin_utc']=ahora();save()
            if estado['estado']=='fallida':raise RuntimeError(estado['motivo'])
        sesion['estado']='finalizada'
        m['estado']='ejecucion_finalizada'
    except KeyboardInterrupt:
        sesion['estado']='interrumpida';m['estado']='interrumpido'
        print('\nInterrumpido. Reanudar con --resume '+str(output),flush=True)
    except Exception as exc:
        sesion['estado']='fallida';m['estado']='fallido';m['error']=str(exc)
        raise
    finally:
        sesion['fin_utc']=ahora();save()
        from graficar import generar
        generar(output)
    print(f'\nResultados: {output}. Consultar estado_configuraciones.csv y cobertura.json.')


if __name__=='__main__':
    main()
