"""Generar borrador PDF desde evidencia REAL; completar interpretacion y firmas."""
import argparse
from html import escape
import json
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',required=True,type=Path)
    p.add_argument('--validation',required=True,type=Path)
    p.add_argument('--deployment',required=True,type=Path)
    p.add_argument('--metadata',type=Path,default=Path('docs/project_metadata.json'))
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    summary=json.loads(a.summary.read_text(encoding='utf-8'))
    validation=json.loads(a.validation.read_text(encoding='utf-8'))
    deployment=json.loads(a.deployment.read_text(encoding='utf-8'))
    metadata=json.loads(a.metadata.read_text(encoding='utf-8'))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet();story=[]
    font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    bold=font.with_name('DejaVuSans-Bold.ttf')
    if font.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont('DocSans',str(font)))
        pdfmetrics.registerFont(TTFont('DocSansBold',str(bold)))
        for name in styles.byName:
            styles[name].fontName='DocSansBold' if name in ('Title','Heading1','Heading2','Heading3') else 'DocSans'
    else:
        import reportlab
        bundled=Path(reportlab.__file__).parent/'fonts'
        pdfmetrics.registerFont(TTFont('DocSans',str(bundled/'Vera.ttf')))
        pdfmetrics.registerFont(TTFont('DocSansBold',str(bundled/'VeraBd.ttf')))
        for name in styles.byName:
            styles[name].fontName='DocSansBold' if name in ('Title','Heading1','Heading2','Heading3') else 'DocSans'
    if not metadata.get('practica3_commit'):
        metadata['practica3_commit']=deployment.get('health',{}).get('body',{}).get('code_commit','')
    def para(t,style='BodyText'):
        story.append(Paragraph(escape(str(t)),styles[style]))
        if not style.startswith('Heading'):
            story.append(Spacer(1,.2*cm))
    para('Practica 03 - Servicio de convolucion con PyCUDA y Docker','Title')
    para('BORRADOR: revisar evidencia y completar analisis, contribuciones y firmas antes de entregar.')
    for k,v in metadata.items():para(f'{k}: {v or "PENDIENTE DE COMPLETAR"}')
    para('Relacion con la practica 02 y arquitectura','Heading1')
    para('Se reutilizan sin modificaciones los kernels propios, float32, extension con ceros y un pixel por hilo en X. '
         'Navegador -> Flask/Gunicorn -> cola FIFO de capacidad 2 -> hilo/contexto PyCUDA -> PNG/descarga. '
         'Un trabajador GPU serializa calculos; se admite una solicitud activa y una pendiente. Exceso: HTTP 429.')
    para('Integracion y alcances de medicion','Heading1')
    para('CPU-GPU y GPU-CPU: memoria NumPy paginable, copias sincronas y perf_counter. Kernel: eventos CUDA '
         'con sincronizacion. Servidor: decodificacion, reserva/liberacion, transferencias, kernel, codificacion y escritura. '
         'Espera de cola separada. Cliente: POST mas descarga completa del PNG. Contexto/compilacion fuera de series estables; '
         'ver gpu_init.json y health de la primera ejecucion. La cache de SourceModule puede estar activa.')
    para('Entorno y despliegue','Heading1')
    for key in ['location','network_note','url']:para(f'{key}: {deployment.get(key,"no registrado")}')
    health=deployment.get('health',{}).get('body',{})
    para(json.dumps(health.get('gpu',{}),ensure_ascii=False))
    para('Validacion','Heading1');para(f"Validacion GPU real aprobada: {validation.get('passed')}. Casos: {len(validation.get('tests',[]))}. Tolerancia: {validation.get('tolerance')}.")
    tests=validation.get('tests',[])
    if tests:
        para('Error maximo directo-servicio: '+str(max(t['direct_service']['error_max'] for t in tests)))
        para('Error maximo CPU-directo: '+str(max(t['cpu_direct']['error_max'] for t in tests)))
    for visual in [a.validation.parent/'original.png']+sorted(a.validation.parent.glob('*_resultado.png')):
        if visual.is_file():
            para(visual.stem,'Heading3')
            im=Image(str(visual));ratio=min(10*cm/im.imageWidth,8*cm/im.imageHeight)
            im.drawWidth=im.imageWidth*ratio;im.drawHeight=im.imageHeight*ratio;story.append(im)
    para('Mediciones','Heading1')
    for group in summary['groups']:
        desc=' / '.join(str(group.get(k,'')) for k in ['mode','experiment','filter','width','height','mask','concurrency'])
        para(desc,'Heading3')
        para(f"n={group['n']}; completa={group['complete']}")
        for name in ['kernel_ms','server_processing_ms','queue_ms','client_total_ms','processing_ms','successful_per_second']:
            if name in group:para(f"{name}: mediana {group[name]['median']:.4f}; IQR {group[name]['iqr']:.4f}")
    para('Errores e incompletas','Heading1');para(json.dumps(summary['failures'],ensure_ascii=False) or '[]')
    for fig in sorted(a.summary.parent.glob('*.png')):
        story.append(PageBreak());para(fig.stem,'Heading1')
        im=Image(str(fig));ratio=min(17*cm/im.imageWidth,22*cm/im.imageHeight)
        im.drawWidth=im.imageWidth*ratio;im.drawHeight=im.imageHeight*ratio;story.append(im)
    story.append(PageBreak());para('Analisis, conclusiones y limitaciones: completar','Heading1')
    for q in ['Equivalencia numerica y efectos exclusivos de abs/clip/redondeo PNG.',
              'Distribucion del tiempo entre GPU, preparacion, servicio y comunicacion.',
              'Comparacion host/contenedor/servicio: identificar si se mantuvo GPU, hash e hilos.',
              'Efecto de resolucion, mascara y concurrencia sobre latencia, cola y rendimiento.',
              'Restricciones de equipo/red y justificacion local/nube; limitaciones documentadas.',
              'Conclusion sustentada: no asumir que Docker acelera kernels.']:
        para(q+' [Redactar con los datos anteriores]')
    para('Contribuciones y firmas','Heading1');para('Integrante 1: __________________  Aporte: __________________  Firma: __________________')
    para('Integrante 2: __________________  Aporte: __________________  Firma: __________________')
    SimpleDocTemplate(str(a.out),pagesize=(21*cm,29.7*cm),leftMargin=2*cm,rightMargin=2*cm).build(story)
    print(a.out)

if __name__=='__main__':main()
