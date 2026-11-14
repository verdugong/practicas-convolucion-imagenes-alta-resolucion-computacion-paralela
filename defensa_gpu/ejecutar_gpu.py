"""Elegir CUDA C o PyCUDA y una imagen nueva, sin ejecutar convolucion CPU."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def cargar_gris(path):
    with Image.open(path) as original:
        im=ImageOps.exif_transpose(original)
        if im.mode not in ('1','L','LA','P','RGB','RGBA','CMYK'):
            raise ValueError('Se requiere imagen de 8 bits: JPG, PNG, PGM, BMP o TIFF compatible.')
        if im.mode in ('LA','P','RGBA'):
            rgba=im.convert('RGBA')
            im=Image.alpha_composite(Image.new('RGBA',rgba.size,(255,255,255,255)),rgba).convert('RGB')
        im=im.convert('L')
        if im.width*im.height>100_000_000:
            raise ValueError('La imagen supera el limite del proyecto: 100 millones de pixeles.')
        return im.copy()


def preparar(gris,tamano):
    if tamano is None:
        return gris.copy(),dict(metodo='Sin recorte ni redimensionamiento',caja=[0,0,*gris.size])
    # Misma region cuadrada para todo el experimento de resolucion.
    w,h=gris.size;lado=min(w,h);x=(w-lado)//2;y=(h-lado)//2
    caja=(x,y,x+lado,y+lado)
    return gris.crop(caja).resize((tamano,tamano),Image.Resampling.LANCZOS),dict(
        metodo='Recorte cuadrado central comun + Pillow LANCZOS',caja=list(caja),lado_salida=tamano)


def construir_plan(a):
    n=a.tamano
    if a.prueba=='individual':
        if a.filtro!='media' and a.mascara!=3:
            raise ValueError('Nitidez y laplaciano solo admiten mascara 3.')
        return [dict(filtro=a.filtro,k=a.mascara,hilos=a.hilos,tamano=n)]
    if a.prueba=='filtros':
        return [dict(filtro=f,k=3,hilos=a.hilos,tamano=n or 4000)
                for f in ('media','nitidez','laplaciano')]
    if a.prueba=='resolucion':
        if n is not None:raise ValueError('--prueba resolucion ya usa 2000, 4000 y 6000; omitir --tamano.')
        return [dict(filtro='media',k=9,hilos=a.hilos,tamano=s) for s in (2000,4000,6000)]
    if a.prueba=='mascara':
        return [dict(filtro='media',k=k,hilos=a.hilos,tamano=n or 4000) for k in (3,9,21)]
    return [dict(filtro='media',k=9,hilos=t,tamano=n or 4000) for t in (128,256,512)]


def estadisticas(valores):
    a=np.asarray(valores,dtype=np.float64)
    if not np.isfinite(a).all() or np.any(a<0):raise ValueError('Tiempos no validos.')
    med=float(np.median(a))
    if a.size<5:return dict(n=int(a.size),mediana=med,q1=None,q3=None,iqr=None)
    q1,q3=np.quantile(a,[.25,.75],method='linear')
    return dict(n=int(a.size),mediana=med,q1=float(q1),q3=float(q3),iqr=float(q3-q1))


def guardar_visual(prefix,salida,filtro):
    if not np.isfinite(salida).all():raise ValueError('Salida GPU con NaN/Inf.')
    valores=np.abs(salida) if filtro=='laplaciano' else salida
    Image.fromarray(np.floor(np.clip(valores,0,255)+.5).astype(np.uint8)).save(str(prefix)+'_resultado.png')


def ejecutar_pycuda(modulo,entrada,c,reps,prefix):
    from src.pycuda.filtros_gpu import crear_filtro
    inicio=perf_counter()
    gpu=crear_filtro(modulo,entrada,c['filtro'],c['k'],c['hilos'])
    reservas_ms=(perf_counter()-inicio)*1000
    rows=[]
    path=Path(str(prefix)+'_raw.csv')
    columnas=['implementacion','filtro','ancho','alto','mascara','hilos','bloques','repeticion',
              'h2d_ms','kernel_ms','d2h_ms','total_ms','unidad','estado','validacion_cpu']
    try:
        print('Calentamiento PyCUDA (1 ejecucion)',flush=True)
        gpu.ejecutar()
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=columnas);w.writeheader();f.flush()
            for r in range(1,reps+1):
                t=gpu.ejecutar()
                if not np.isfinite(gpu.salida).all():raise ValueError('Resultado GPU no finito.')
                row=dict(implementacion='PYCUDA',filtro=c['filtro'],ancho=entrada.shape[1],
                    alto=entrada.shape[0],mascara=c['k'],hilos=c['hilos'],bloques=gpu.bloques,
                    repeticion=r,**t,unidad='ms',estado='completada',validacion_cpu='no_realizada')
                w.writerow(row);f.flush();rows.append(row)
                print(f"PyCUDA {r}/{reps}: kernel={t['kernel_ms']:.6f} ms, total={t['total_ms']:.6f} ms",flush=True)
        salida=gpu.salida.copy()
        salida.tofile(str(prefix)+'_salida.f32')
        return rows,salida,dict(reservas_ms=reservas_ms,reservas_incluidas_en_total=False)
    finally:gpu.cerrar()


def ejecutar_cuda(exe,pgm,c,reps,prefix,forma):
    cmd=[str(exe),'--input',str(pgm),'--filter',c['filtro'],'--mask',str(c['k']),
         '--threads',str(c['hilos']),'--reps',str(reps),'--output',str(prefix)]
    print('Ejecutable CUDA C independiente:',exe,flush=True)
    r=subprocess.run(cmd,capture_output=True,text=True,errors='replace')
    Path(str(prefix)+'_consola.txt').write_text(r.stdout+r.stderr,encoding='utf-8')
    print(r.stdout,end='',flush=True)
    if r.returncode:raise RuntimeError(r.stderr or r.stdout)
    with Path(str(prefix)+'_raw.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    raw=Path(str(prefix)+'_salida.f32')
    if raw.stat().st_size!=int(np.prod(forma))*4:raise ValueError('Salida binaria CUDA incompleta.')
    salida=np.fromfile(raw,dtype='<f4').reshape(forma)
    return rows,salida,dict(comando=cmd)


def ejecutar_caso(a,c,gris,out,modulo=None,exe=None):
    imagen,preparacion=preparar(gris,c['tamano'])
    nombre=f"{c['filtro']}_{imagen.width}x{imagen.height}_k{c['k']}_t{c['hilos']}"
    folder=out/nombre;folder.mkdir()
    pgm=folder/'entrada.pgm';imagen.save(pgm)
    imagen.save(folder/'original_gris.png')
    entrada=np.array(imagen,dtype=np.float32,order='C')
    prefix=folder/a.version
    print(f"\n{a.version.upper()} | {c['filtro']} {c['k']}x{c['k']} | "
          f"{imagen.width}x{imagen.height} | {c['hilos']} hilos",flush=True)
    print(preparacion['metodo'],flush=True)
    if a.version=='pycuda':
        rows,salida,extra=ejecutar_pycuda(modulo,entrada,c,a.reps,prefix)
    else:
        rows,salida,extra=ejecutar_cuda(exe,pgm,c,a.reps,prefix,entrada.shape)
    if len(rows)!=a.reps:raise ValueError('Faltan repeticiones; no se resume como prueba completa.')
    guardar_visual(prefix,salida,c['filtro'])
    resumen=dict(configuracion=nombre,version=a.version,ancho=imagen.width,alto=imagen.height,
        filtro=c['filtro'],mascara=c['k'],hilos=c['hilos'],repeticiones=a.reps,
        validacion_cpu='no_realizada',aceleracion=None,estado='completada',
        preparacion=preparacion,sha256_pgm=sha256(pgm),**extra)
    for met in ('h2d_ms','kernel_ms','d2h_ms','total_ms'):
        resumen[met]=estadisticas([float(r[met]) for r in rows])
    print(f"Mediana kernel: {resumen['kernel_ms']['mediana']:.6f} ms; "
          f"total: {resumen['total_ms']['mediana']:.6f} ms",flush=True)
    (folder/'resumen.json').write_text(json.dumps(resumen,ensure_ascii=False,indent=2),encoding='utf-8')
    return resumen


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',required=True,choices=['cuda','pycuda'])
    p.add_argument('--imagen',required=True,type=Path,help='Imagen nueva JPG, PNG, PGM, BMP o TIFF de 8 bits')
    p.add_argument('--prueba',choices=['individual','filtros','resolucion','mascara','hilos'],default='individual')
    p.add_argument('--filtro',choices=['media','nitidez','laplaciano'],default='media')
    p.add_argument('--mascara',type=int,choices=[3,9,21],default=3)
    p.add_argument('--hilos',type=int,choices=[128,256,512],default=256)
    p.add_argument('--reps',type=int,default=5,help='1 para demostracion rapida; 5 o mas para estadistica experimental')
    p.add_argument('--tamano',type=int,help='Opcional: recorte central cuadrado y redimensionamiento a N x N')
    p.add_argument('--salida',type=Path,help='Carpeta NUEVA; si se omite, usa results/defensa_gpu con fecha y hora')
    return p


def main(argv=None):
    p=parser();a=p.parse_args(argv)
    if not 1<=a.reps<=1000:p.error('--reps debe estar entre 1 y 1000.')
    if a.tamano is not None and not 1<=a.tamano<=10000:p.error('--tamano debe estar entre 1 y 10000.')
    if not a.imagen.is_file():p.error('No existe la imagen: '+str(a.imagen))
    if not (ROOT/'src/pycuda/filtros_gpu.py').is_file():p.error('Colocar defensa_gpu junto a src e include.')
    try:plan=construir_plan(a)
    except ValueError as exc:p.error(str(exc))
    exe=HERE/'bin'/('cuda_individual.exe' if sys.platform=='win32' else 'cuda_individual')
    if a.version=='cuda' and not exe.is_file():p.error('Compilar antes con python defensa_gpu/compilar_cuda.py')
    gris=cargar_gris(a.imagen)
    out=(a.salida or ROOT/'results/defensa_gpu'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    out.mkdir(parents=True,exist_ok=False)
    m=dict(inicio_utc=datetime.now(timezone.utc).isoformat(),version=a.version,prueba=a.prueba,
        imagen=str(a.imagen.resolve()),sha256_imagen=sha256(a.imagen),dimensiones_gris=list(gris.size),
        numpy=np.__version__,pillow=PILLOW_VERSION,plan=plan,estado='en_curso',resultados=[],
        validacion_cpu='no_realizada: demostracion exclusivamente GPU',
        alcance='CPU anfitrion prepara/copia/guarda; convolucion solo en GPU. Un calentamiento y N repeticiones. '
                'Reservas/contexto/compilacion/imagenes excluidos del total GPU. Sin speedup por faltar referencia CPU.')
    if a.reps<5:print('DEMOSTRACION: menos de 5 repeticiones; no sustituye los experimentos del informe.',flush=True)
    contexto=None;modulo=None
    try:
        if a.version=='pycuda':
            import pycuda
            import pycuda.driver as cuda
            from src.pycuda.filtros_gpu import crear_modulo
            t0=perf_counter();cuda.init();contexto=cuda.Device(0).make_context()
            m['contexto_inicializacion_ms']=(perf_counter()-t0)*1000
            t0=perf_counter();modulo=crear_modulo()
            m['compilacion_carga_ms']=(perf_counter()-t0)*1000
            m['gpu']=cuda.Device(0).name();m['pycuda']=pycuda.VERSION_TEXT
            m['driver_api']=cuda.get_driver_version()
            m['cache_compilador']='SourceModule usa su cache por defecto; compilacion/carga pueden reutilizarla.'
        else:m['ejecutable_sha256']=sha256(exe)
        for c in plan:
            m['resultados'].append(ejecutar_caso(a,c,gris,out,modulo,exe))
        m['estado']='completada'
    except KeyboardInterrupt:
        m['estado']='interrumpida';raise
    except Exception as exc:
        m['estado']='fallida';m['error']=str(exc);raise
    finally:
        modulo=None
        if contexto is not None:contexto.pop();contexto.detach()
        (out/'ejecucion.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    # Resumen facil de abrir para una serie de filtros/resoluciones/mascaras/hilos.
    with (out/'resumen.csv').open('w',newline='',encoding='utf-8') as f:
        cols=['configuracion','version','ancho','alto','filtro','mascara','hilos','repeticiones',
              'kernel_mediana_ms','kernel_iqr_ms','total_mediana_ms','total_iqr_ms','validacion_cpu']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in m['resultados']:
            w.writerow({**{k:r[k] for k in cols if k in r},
                'kernel_mediana_ms':r['kernel_ms']['mediana'],'kernel_iqr_ms':r['kernel_ms']['iqr'],
                'total_mediana_ms':r['total_ms']['mediana'],'total_iqr_ms':r['total_ms']['iqr']})
    print('\nResultados:',out,flush=True)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('\nDemostracion interrumpida; resultados previos conservados.',file=sys.stderr);sys.exit(130)
    except (OSError,RuntimeError,ValueError) as exc:
        print('ERROR:',exc,file=sys.stderr);sys.exit(1)
