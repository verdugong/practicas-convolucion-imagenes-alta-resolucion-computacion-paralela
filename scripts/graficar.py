"""Consolidar datos parciales sin inventar mediciones ni aceleraciones incompletas."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from comun import guardar_json, leer_filas, sha256

IMPLEMENTACIONES=('CPU','CUDA','PYTHON','PYCUDA')
NOMBRES={'CPU':'C++ secuencial','CUDA':'CUDA C','PYTHON':'Python secuencial','PYCUDA':'PyCUDA'}
COLORES={'CPU':'#475569','CUDA':'#1769aa','PYTHON':'#16875c','PYCUDA':'#d97706'}
ROOT=Path(__file__).resolve().parents[1]


def estadisticas(valores):
    q1,med,q3=np.quantile(valores,[.25,.5,.75],method='linear')
    return dict(mediana=float(med),q1=float(q1),q3=float(q3),iqr=float(q3-q1))


def calcular_aceleraciones(medianas):
    """Unico calculo del consolidado: C++/CUDA C y Python/PyCUDA.

    medianas contiene SOLO versiones completas, validadas y comparables.
    No dividir por hilos/bloque: esa cantidad no es p de eficiencia paralela.
    """
    resultado={}
    for gpu,base in [('CUDA','CPU'),('PYCUDA','PYTHON')]:
        if gpu in medianas and base in medianas:
            resultado[gpu]=dict(referencia_aceleracion=base,
                aceleracion_kernel=medianas[base]['total_ms']/medianas[gpu]['kernel_ms'],
                aceleracion_total=medianas[base]['total_ms']/medianas[gpu]['total_ms'])
    if 'CUDA' in medianas and 'PYCUDA' in medianas:
        resultado['comparacion_gpu']=dict(
            razon_cuda_pycuda_kernel=medianas['CUDA']['kernel_ms']/medianas['PYCUDA']['kernel_ms'],
            razon_cuda_pycuda_total=medianas['CUDA']['total_ms']/medianas['PYCUDA']['total_ms'])
    return resultado


def archivos_version(root,c,impl):
    etapa='CPU_CUDA' if impl in ('CPU','CUDA') else impl
    suffix={'CPU':'_raw.csv','CUDA':'_raw.csv','PYTHON':'_python_raw.csv','PYCUDA':'_pycuda_raw.csv'}[impl]
    if 'etapas' in c:
        e=c['etapas'].get(etapa,{})
        if 'prefijo' not in e:return None,e
        return Path(str(root/e['prefijo'])+suffix),e
    return root/(c['prefijo']+suffix),{}


def validar_fila(r,c,impl,legacy=False):
    try:
        if (r['implementacion']!=impl or r['filtro']!=c['filtro']
            or int(r['ancho'])!=c['resolucion'] or int(r['alto'])!=c['resolucion']
            or int(r['mascara'])!=c['mascara'] or r['unidad']!='ms'
            or (not legacy and r.get('estado')!='completada')):return False
        tiempos=['total_ms'] if impl in ('CPU','PYTHON') else ['total_ms','kernel_ms','h2d_ms','d2h_ms']
        for key in tiempos:
            v=float(r[key])
            if not np.isfinite(v) or v<0 or (key in ('total_ms','kernel_ms') and v==0):return False
        if impl in ('CUDA','PYCUDA'):
            if int(r['hilos'])!=c['hilos'] or int(r['bloques'])!=(c['resolucion']**2+c['hilos']-1)//c['hilos']:return False
            if float(r['h2d_ms'])+float(r['d2h_ms'])>float(r['total_ms'])*1.00001:return False
        if impl!='CPU':
            tol=float(r['tolerancia'])
            if not 0<=tol<=.001 or r['validado']!='true':return False
            for key in ['error_max','error_medio']:
                v=float(r[key])
                if not np.isfinite(v) or not 0<=v<=tol:return False
            if impl=='PYCUDA':
                if r['validado_cuda']!='true':return False
                for key in ['error_cuda_max','error_cuda_medio']:
                    v=float(r[key])
                    if not np.isfinite(v) or not 0<=v<=tol:return False
        return True
    except (KeyError,TypeError,ValueError,OverflowError):return False


def resumir(root):
    root=Path(root)
    m=json.loads((root/'experimentos.json').read_text(encoding='utf-8'))
    reps=m['repeticiones'];legacy=m.get('formato')!=2
    if reps<5:raise ValueError('Minimo cinco repeticiones.')
    resumen,raw,estados=[],[],[]
    for c in m['configuraciones']:
        medianas={};registros=[]
        requeridas=c.get('requeridas',list(IMPLEMENTACIONES) if c['experimentos']!=['hilos'] else ['CUDA','PYCUDA'])
        for impl in IMPLEMENTACIONES:
            path,e=archivos_version(root,c,impl)
            filas=[r for r in leer_filas(path) if r.get('implementacion')==impl] if path else []
            validas=all(validar_fila(r,c,impl,legacy) for r in filas)
            indices=[int(r.get('repeticion',-1)) for r in filas]
            completas=(len(filas)==reps and set(indices)==set(range(1,reps+1)) and validas
                        and (legacy or e.get('estado')=='completada'))
            if completas:estado='completada'
            elif filas and (not validas or len(indices)!=len(set(indices)) or len(filas)>reps):estado='invalida'
            elif impl not in requeridas:estado='no_requerida'
            elif e.get('estado')=='fallida':estado='fallida'
            elif filas or e.get('estado') in ('no_completada','en_curso'):estado='no_completada'
            else:estado='pendiente'
            st=dict(configuracion=c['prefijo'],implementacion=impl,estado=estado,
                requerida=impl in requeridas,repeticiones=len(filas),objetivo=reps,
                motivo=e.get('motivo',''),limite_s=e.get('limite_s',''),tiempo_proceso_s='',
                fase_interrumpida='',fase_transcurrido_s='',piloto_estimacion_s='')
            if e.get('intentos'):
                st['tiempo_proceso_s']=e['intentos'][-1].get('tiempo_proceso_s','')
            if impl=='PYTHON' and path:
                estado_path=Path(str(path).replace('_python_raw.csv','_python_estado.json'))
                if estado_path.exists():
                    detalle=json.loads(estado_path.read_text(encoding='utf-8'))
                    st['fase_interrumpida']=detalle.get('fase','')
                    st['fase_transcurrido_s']=detalle.get('fase_incompleta_transcurrido_s','')
                    st['piloto_estimacion_s']=detalle.get('piloto',{}).get('estimacion_por_repeticion_s','')
            estados.append(st)
            raw.extend(dict(configuracion=c['prefijo'],estado_configuracion=estado,
                estado=r.get('estado','completada' if validar_fila(r,c,impl,True) else 'invalida'),
                **{k:v for k,v in r.items() if k!='estado'}) for r in filas)
            if not completas:continue
            registro=dict(configuracion=c['prefijo'],implementacion=impl,filtro=c['filtro'],
                resolucion=c['resolucion'],mascara=c['mascara'],hilos=c['hilos'] if impl in ('CUDA','PYCUDA') else 0,
                experimentos=';'.join(c['experimentos']),repeticiones=reps,estado='completada',
                referencia_aceleracion='',aceleracion_kernel='',aceleracion_total='',
                eficiencia_paralela='',motivo_eficiencia='No aplica S/p: CPU y GPU heterogeneas; hilos/bloque no es p')
            campos=['total_ms'] if impl in ('CPU','PYTHON') else ['h2d_ms','kernel_ms','d2h_ms','total_ms']
            medianas[impl]={}
            for campo in campos:
                stats=estadisticas([float(r[campo]) for r in filas])
                registro.update({campo+'_'+k:v for k,v in stats.items()})
                medianas[impl][campo]=stats['mediana']
            if impl!='CPU':
                registro['error_max']=max(float(r['error_max']) for r in filas)
                registro['error_medio_peor']=max(float(r['error_medio']) for r in filas)
            if impl in ('CUDA','PYCUDA'):
                registro['transferencias_pct_mediana']=estadisticas([
                    100*(float(r['h2d_ms'])+float(r['d2h_ms']))/float(r['total_ms']) for r in filas])['mediana']
                registro['fraccion_kernel_pct_mediana']=estadisticas([
                    100*float(r['kernel_ms'])/float(r['total_ms']) for r in filas])['mediana']
            registros.append(registro)
        aceleraciones=calcular_aceleraciones(medianas)
        for r in registros:
            r.update(aceleraciones.get(r['implementacion'],{}))
            if r['implementacion']=='PYCUDA':r.update(aceleraciones.get('comparacion_gpu',{}))
        resumen.extend(registros)
    return m,resumen,raw,estados


def escribir_csv(path,rows,columnas=None):
    columnas=columnas or list(dict.fromkeys(k for r in rows for k in r)) or ['estado']
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columnas);w.writeheader();w.writerows(rows)


def cobertura(m,estados):
    indice={(r['configuracion'],r['implementacion']):r for r in estados}
    alternativas={(c['filtro'],c['mascara']):c for c in m['configuraciones'] if c.get('alternativa')}
    faltantes=[];excepciones=[]
    for c in m['configuraciones']:
        req=c.get('requeridas',list(IMPLEMENTACIONES) if c['experimentos']!=['hilos'] else ['CUDA','PYCUDA'])
        for impl in req:
            r=indice[c['prefijo'],impl]
            if r['estado']=='completada':continue
            alt=alternativas.get((c['filtro'],c['mascara']))
            alterna_ok=alt and all(indice[alt['prefijo'],v]['estado']=='completada' for v in IMPLEMENTACIONES)
            evidencia=(r['estado']=='no_completada' and r['motivo'] and
                       isinstance(r['tiempo_proceso_s'],(int,float)) and r['tiempo_proceso_s']>0)
            if impl=='PYTHON' and not c.get('alternativa') and alterna_ok and evidencia:
                excepciones.append(c['prefijo'])
            else: faltantes.append(c['prefijo']+'/'+impl+': '+r['estado'])
    return dict(cobertura_experimental_satisfecha=not faltantes,
        configuraciones_python_con_limitacion_documentada=excepciones,pendientes=faltantes,
        advertencia='Verifica cobertura de mediciones; no certifica informe, defensa, procedencia ni commits.')


def graficas(out,m,rows):
    figuras=out/'figuras';figuras.mkdir(exist_ok=True)
    indice={(r['configuracion'],r['implementacion']):r for r in rows}
    for grupo,key,titulo in [('filtros','filtro','Filtros 3x3 / 4000x4000'),
            ('resoluciones','resolucion','Media 9x9 / resolucion'),
            ('mascaras','mascara','Media / mascaras / 4000x4000'),
            ('hilos','hilos','Media 9x9 / hilos / 4000x4000'),
            ('alternativa','prefijo','Comparacion adicional a resolucion comun')]:
        configs=[c for c in m['configuraciones'] if grupo in c['experimentos']]
        if not configs:continue
        if key in ('resolucion','mascara','hilos'):configs.sort(key=lambda c:c[key])
        labels=[(f"{c['filtro']} k{c['mascara']}" if grupo=='alternativa' else str(c[key])) for c in configs]
        fig,axes=plt.subplots(1,2,figsize=(12,4.7))
        for ax,metric,impls in [(axes[0],'total_ms',('CUDA','PYCUDA') if grupo=='hilos' else IMPLEMENTACIONES),
                              (axes[1],'kernel_ms',('CUDA','PYCUDA'))]:
            cuantos=0
            for j,impl in enumerate(impls):
                puntos=[(i,indice.get((c['prefijo'],impl))) for i,c in enumerate(configs)]
                puntos=[(i,r) for i,r in puntos if r is not None]
                if not puntos:continue
                xx=[i+(j-(len(impls)-1)/2)*.09 for i,r in puntos]
                yy=[r[metric+'_mediana'] for i,r in puntos]
                err=[[r[metric+'_mediana']-r[metric+'_q1'] for i,r in puntos],
                     [r[metric+'_q3']-r[metric+'_mediana'] for i,r in puntos]]
                ax.errorbar(xx,yy,yerr=err,fmt='o',capsize=4,label=NOMBRES[impl],color=COLORES[impl]);cuantos+=len(puntos)
            ax.set_xticks(range(len(labels)),labels,rotation=20 if grupo=='alternativa' else 0)
            ax.grid(axis='y',alpha=.25);ax.set_ylabel('ms; mediana y Q1-Q3')
            if cuantos:ax.legend()
            else:ax.text(.5,.5,'Sin mediciones completas',ha='center',transform=ax.transAxes)
            if metric=='total_ms' and cuantos:ax.set_yscale('log')
            ax.set_title('Tiempo total (escala log)' if metric=='total_ms' else 'Kernel GPU')
        fig.suptitle(titulo+' | los casos incompletos no se grafican')
        fig.tight_layout();fig.savefig(figuras/(grupo+'_tiempos.png'),dpi=160);plt.close(fig)
    # Aceleraciones con bases explicitamente diferentes.
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    configs=m['configuraciones'];labels=[c['prefijo'].replace('_','\n',1) for c in configs]
    for ax,met in zip(axes,['aceleracion_kernel','aceleracion_total']):
        for offset,impl,base in [(-.15,'CUDA','C++'),(.15,'PYCUDA','Python')]:
            puntos=[(i,indice.get((c['prefijo'],impl))) for i,c in enumerate(configs)]
            puntos=[(i,r) for i,r in puntos if r and r[met]!='']
            if puntos:ax.bar([i+offset for i,r in puntos],[r[met] for i,r in puntos],.3,label=base+'/'+NOMBRES[impl],color=COLORES[impl])
        ax.set_ylabel(met+' (x)');ax.grid(axis='y',alpha=.25)
        handles,_=ax.get_legend_handles_labels()
        if handles:ax.legend();ax.set_yscale('log')
    axes[1].set_xticks(range(len(labels)),labels,rotation=65,ha='right',fontsize=7)
    fig.suptitle('Aceleraciones: bases C++ y Python; escala log. Sin barra = datos incompletos.')
    fig.tight_layout();fig.savefig(figuras/'aceleraciones.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5))
    for offset,impl in [(-.16,'CUDA'),(.16,'PYCUDA')]:
        puntos=[(i,indice.get((c['prefijo'],impl))) for i,c in enumerate(configs)]
        puntos=[(i,r) for i,r in puntos if r]
        if puntos:
            ax.bar([i+offset for i,r in puntos],
                [r['transferencias_pct_mediana'] for i,r in puntos],.32,
                label=NOMBRES[impl],color=COLORES[impl])
    ax.set_xticks(range(len(labels)),labels,rotation=65,ha='right',fontsize=7)
    ax.set_ylim(0,100);ax.set_ylabel('Mediana de (H2D+D2H)/total (%)')
    ax.set_title('Proporcion temporal de transferencias; no es eficiencia paralela')
    handles,_=ax.get_legend_handles_labels()
    if handles:ax.legend()
    fig.tight_layout();fig.savefig(figuras/'transferencias.png',dpi=160);plt.close(fig)


def comparativos(root,out,m):
    from comparar_imagenes import panel
    for c in m['configuraciones']:
        if c['resolucion']!=4000 or c['hilos']!=256:continue
        dato=m.get('entradas',{}).get('4000',{})
        candidatos=[Path(dato.get('path','__no_existe__')),ROOT/'data/preparadas/imagen_4000.pgm']
        original=next((p for p in candidatos if p.is_file() and sha256(p)==dato.get('sha256')),None)
        if original is None:continue
        rutas=[original];etiquetas=['Original']
        for impl,suf in [('CPU','_cpu.pgm'),('CUDA','_gpu.pgm'),('PYTHON','_python.pgm'),('PYCUDA','_pycuda.pgm')]:
            path,e=archivos_version(root,c,impl)
            if path is None:continue
            csvsuffix={'CPU':'_raw.csv','CUDA':'_raw.csv','PYTHON':'_python_raw.csv','PYCUDA':'_pycuda_raw.csv'}[impl]
            visual=Path(str(path)[:-len(csvsuffix)]+suf)
            if visual.exists() and (m.get('formato')!=2 or e.get('estado')=='completada'):
                rutas.append(visual);etiquetas.append(NOMBRES[impl])
        if len(rutas)<2:continue
        ims=[]
        for p in rutas:
            with Image.open(p) as im:ims.append(im.convert('L'))
        if len({im.size for im in ims})!=1:continue
        rec=m.get('recorte',dict(x=1600,y=600,lado=600));x,y,lado=rec['x'],rec['y'],rec['lado']
        panel(ims,etiquetas,True).save(out/'figuras'/f"{c['prefijo']}_completa.png")
        panel([im.crop((x,y,x+lado,y+lado)) for im in ims],etiquetas,False).save(out/'figuras'/f"{c['prefijo']}_detalle.png")


def generar(root,salida=None):
    root=Path(root);out=Path(salida) if salida else root;out.mkdir(parents=True,exist_ok=True)
    m,resumen,raw,estados=resumir(root)
    escribir_csv(out/'resumen.csv',resumen)
    escribir_csv(out/'mediciones.csv',raw)
    escribir_csv(out/'estado_configuraciones.csv',estados)
    guardar_json(out/'cobertura.json',cobertura(m,estados))
    graficas(out,m,resumen);comparativos(root,out,m)
    print(f'Consolidado: {len(resumen)} versiones/configuraciones completas. Estado y pendientes: {out}')
    return resumen,estados


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('resultados',type=Path)
    p.add_argument('--out',type=Path,help='Destino aparte; util para no modificar evidencia original')
    a=p.parse_args();generar(a.resultados,a.out)


if __name__=='__main__':main()
