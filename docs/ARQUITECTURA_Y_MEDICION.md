# Arquitectura, continuidad y tiempos

## Qué se conservó y qué cambió

`src/pycuda/*`, `src/python/*`, fuentes C++/CUDA C, include, scripts y tests de la
práctica 02 permanecen byte a byte. `tools.verify_sources` comprueba sus SHA-256.
Los kernels propios no se sustituyen por OpenCV/SciPy ni filtros NumPy. Se mantiene
float32, `--fmad=false`, extensión con ceros, media 1/k², nitidez
[0,-1,0;-1,5,-1;0,-1,0], laplaciano [0,1,0;1,-4,1;0,1,0], un píxel por hilo y
bloques distribuidos en X. La referencia secuencial solo se usa en matrices
pequeñas de validación; el servicio no recurre a CPU ante falta de NVIDIA.

Nuevo: webapp contiene interfaz Flask y adaptador de los filtros originales;
tools contiene CLI directa, cliente de experimentos, validación real, captura
Docker, consolidación y borrador PDF. El README previo se guarda en
`docs/antecedentes/README.md`; los .bat anteriores continúan siendo de práctica 02.
Los experimentos nuevos no usan la campaña anterior como si fuese del servicio.

## Contexto, memoria y concurrencia

Gunicorn crea un proceso, sin preload, con 8 hilos HTTP. GPUWorker inicia después
del fork un hilo exclusivo que importa PyCUDA, llama cuda.init(), crea
Device(0).make_context() y ejecuta crear_modulo(). Registra ambos intervalos.
SourceModule puede usar su caché: compile_load_ms no siempre es compilación completa.
No se usa pycuda.autoinit. El contexto queda actual exclusivamente en el hilo GPU.
Todas las reservas, ejecuciones y liberaciones ocurren en ese mismo hilo.
Al cerrar se elimina el módulo y se hace pop/detach en el hilo propietario.

Por solicitud se crean las clases originales con dos buffers de dispositivo y
salida NumPy. Se libera memoria de GPU en finally al terminar el cálculo; no se
reutilizan buffers entre solicitudes. En OOM se emite HTTP 507. Un error de
contexto CUDA fatal puede requerir reiniciar el contenedor; conservar el log.
Las dos solicitudes concurrentes se admiten y se calculan secuencialmente en
FIFO; la segunda acumula queue_ms. Esto evalúa concurrencia HTTP, no dos kernels
simultáneos. Un semáforo limita activa+pendiente a dos y rechaza exceso con 429.

## Alcances exactos (todos los tiempos en ms)

| Campo | Medición y alcance |
|---|---|
| context_init_ms | cuda.init y creación de contexto; importaciones anteriores fuera |
| compile_load_ms | crear_modulo/SourceModule; puede reutilizar caché |
| allocation_ms | creación de clase, buffers GPU, salida NumPy, eventos y argumentos |
| h2d_ms | perf_counter alrededor de copia síncrona y sincronización CUDA |
| kernel_ms | eventos CUDA, evento final sincronizado antes de consultar |
| d2h_ms | perf_counter alrededor de copia síncrona al host |
| total_ms | H2D + lanzamiento/espera + D2H en clase original; no reservas |
| decode_prepare_ms | Pillow, verificación, EXIF/transparencia/gris, conversión float32 |
| encode_ms | preparación PNG de resultado y entrada; NPY si raw=1 |
| processing_ms | decodificar + reservas/liberación + GPU + comprobación finita + PNG/hashes |
| write_ms | escritura de resultado, original y opcional NPY |
| server_processing_ms | trabajo GPUWorker completo hasta finalizar escritura; sin cola ni envío HTTP |
| queue_ms | desde encolar hasta iniciar el trabajo; no incluye subida del formulario |
| client_total_ms | desde POST hasta recibir PNG completo; subida, espera, JSON y descarga incluidas |
| round_ms | desde lanzar ronda hasta completar todas las solicitudes; reloj del mismo cliente |
| successful_per_second | solicitudes exitosas / segundos de la ronda, no suma de tiempos individuales |

Memoria de host paginable NumPy, copias síncronas; no se declara memoria fijada.
No deben sumarse campos anidados como processing_ms + server_processing_ms.
El tiempo de JSON/metadatos y de envío HTTP aparece en latencia cliente pero no
en server_processing_ms. El cliente lee/identifica la imagen antes de cronometrar.
No hay carga de contexto ni compilación en cada solicitud estable.

Cada configuración registra una primera llamada por separado (`cold_first`), un
calentamiento y al menos cinco repeticiones estables. La primera llamada de cada
configuración NO implica nuevo contexto frío. Para arranque en frío del proceso,
reiniciar contenedor y conservar health/context_init_ms/compile_load_ms y la
primera solicitud/log. Su alcance excluye descarga de imagen Docker y arranque
completo del SO; la caché del compilador puede existir. Las primeras llamadas
posteriores se conservan como primeras del caso, no como nuevos arranques fríos.

## Diseño experimental

7 configuraciones únicas: media9 con 2000/4000/6000; media3/media21/nitidez3/
laplaciano3 con 4000. Media9/4000 se comparte entre integración, resolución y
máscaras porque tiene idénticos factores. Hilos fijos 256 por defecto; todas las
herramientas aceptan 128/512 si se conserva igual valor en los tres modos.
Concurrencia: media9/4000, c=1 y c=2, cinco rondas, una solicitud por cliente y
barrera de salida. Cada ronda reporta su propio rendimiento y cada petición su
latencia/cola. Se conserva cualquier 429/error, sin reintentos que los oculten.

La comparación usa los mismos archivos PGM y hashes del ZIP. Python 3.14 y el
freeze Windows anterior no se copian como entorno Linux: la imagen fija Ubuntu
22.04/Python 3.10, CUDA 12.6.3, PyCUDA 2024.1 y dependencias declaradas. Esta es una
condición de entorno distinta que exige revalidación. Para atribuir diferencias
a Docker, usar host compatible con las mismas versiones, GPU y datos; de lo
contrario discutir conjuntamente versiones/OS o aplicar la excepción del enunciado.
No atribuir diferencias a Docker si solo se midió dentro del contenedor y HTTP.

Sin memoria/tiempo: conservar CSV de error, log, configuración y tiempo transcurrido;
no reemplazar tiempos por el límite. Usar `--small` en direct/client para la misma
imagen_512.pgm incluida y explicar la limitación. El timeout HTTP cliente de 300 s
puede terminar la espera mientras el servidor aún trabaja; esperar o reiniciar
antes de otra serie y documentar. Un kernel en ejecución no se cancela de forma
segura desde Flask; no afirmar que el timeout termina el cálculo.
