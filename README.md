# Practica 02 corregida: C++, CUDA C, Python secuencial y PyCUDA

Este paquete contiene el codigo corregido, las tres entradas, pruebas, ejecucion
reanudable, consolidacion de CSV y graficas. Conserva la evidencia del laboratorio
del 10 de noviembre de 2026 en `evidencia_original/`, junto con el codigo que la produjo.
No contiene informe academico ni presentacion, ni inventa mediciones GPU pendientes.

## 1. Empezar en Windows / RTX 3070

Extraer en una carpeta NUEVA. No copiar carpetas build antiguas.
Abrir **x64 Native Tools Command Prompt for VS 2022** y entrar a esta carpeta.
Se requieren Python de 64 bits >=3.10, Visual Studio con C++, CUDA Toolkit,
CMake >=3.24, una GPU NVIDIA y los paquetes de requirements.txt.

Si ya tienes el Python funcional del laboratorio, utilizarlo. Comprobar:

```bat
where python
where cl
where nvcc
where cmake
python -m pip install -r requirements.txt
compilar_windows.bat
```

El script compila en Release para arquitectura 86 (RTX 3070) y ejecuta **cinco pruebas
CTest**: CPU C++, CPU Python, regresiones del pipeline, CPU/CUDA C y las cuatro
versiones mediante PyCUDA. El registro queda en `verificacion_gpu_laboratorio.txt`.
La prueba de pipeline contiene pruebas internas; sus fixtures temporales no se
incorporan como resultados de laboratorio.

Si prefieres un entorno virtual nuevo, antes de instalar:

```bat
py -m venv .venv
.venv\Scripts\activate
```

PyCUDA puede requerir compilacion durante pip install; ejecutar desde la terminal
de Visual Studio con nvcc disponible. Si falla una dependencia o prueba, conservar
el error y resolverlo antes de tomar tiempos. No sustituir PyCUDA por NumPy.

Las entradas PGM ya estan incluidas: **no ejecutar de nuevo el conversor sobre la
misma carpeta**. No se redimensionaron los originales en la correccion.

## 2. Ejecutar una campana nueva

```bat
python scripts\experimentos.py --exe build\practica1.exe --data data\preparadas --out results\campana_final --reps 5 --python-max-segundos 120
```

Alternativa equivalente con carpeta automatica fechada:

```bat
ejecutar_windows.bat
```

Orden de ejecucion, siempre sin concurrencia:

1. C++/CUDA C y PyCUDA de las nueve configuraciones obligatorias.
2. C++/CUDA C y PyCUDA de cinco configuraciones adicionales a 512x512.
3. Python secuencial completo de esas cinco configuraciones pequenas.
4. Python secuencial de las siete configuraciones grandes que lo requieren.

Las pruebas de hilos de 128 y 512 solo requieren CUDA C y PyCUDA. El ejecutable C++
produce ademas su referencia CPU para validar; esa referencia auxiliar se conserva,
pero no se repite Python ni se incluye CPU en la grafica de hilos.

El presupuesto predeterminado de 120 s se aplica **por proceso Python grande**, e
incluye arranque, preparacion, piloto, calentamiento y repeticiones. Es una decision
operativa explicita de tiempo de laboratorio, no un resultado del algoritmo. Ajustarlo
al tiempo realmente disponible. No impone limite a la comparacion pequena comun.

Antes de cada configuracion Python se mide una muestra de hasta 64x64 y se imprime
una ESTIMACION de viabilidad. Nunca se incorpora como tiempo medido de la imagen
grande. El proceso intenta realmente la configuracion grande hasta terminar o alcanzar
su presupuesto. Se guardan tiempo transcurrido, fase, motivo y repeticiones completas.

Si no completa, queda `no_completada`; no se calcula su aceleracion. La comparacion
alternativa de cuatro versiones a la misma resolucion conserva filtro y mascara, y
las pruebas GPU grandes permanecen disponibles. `cobertura.json` verifica esta
alternativa segun el enunciado; no certifica procedencia, informe, GitHub o defensa.

Para no limitar Python (puede necesitar horas):

```bat
ejecutar_windows.bat --python-max-segundos 0
```

La entrada adicional se obtiene desde imagen_4000.pgm con Pillow LANCZOS. Se registra
su hash y metodo. No reemplaza las tres imagenes obligatorias. Se puede cambiar el
lado al iniciar: `--alternativa-lado 256` (rango 64..1000).

## 3. Interrumpir y reanudar

Usar **Ctrl+C una vez** y esperar a que se guarde el estado y se consoliden las tablas.
Si el motivo sera fin de clase, se puede declarar al iniciar:
`--motivo-interrupcion "Fin del horario de laboratorio"`.

```bat
reanudar_windows.bat results\campana_final
```

O directamente:

```bat
python scripts\experimentos.py --exe build\practica1.exe --data data\preparadas --resume results\campana_final
```

La reanudacion verifica codigo, ejecutable, entradas, equipo, GPU/controlador y
versiones. Tambien verifica hashes de los archivos de las etapas completas. No
mezcla datos si cambian esas condiciones. Las etapas completas se omiten. Python
reanuda las repeticiones completas de su CSV y vuelve a calentar; CUDA reinicia una
etapa interrumpida en otro prefijo, preservando el intento previo.

Los casos que agotaron su presupuesto no se reintentan automaticamente. Para
retomarlos con mas tiempo:

```bat
python scripts\experimentos.py --exe build\practica1.exe --data data\preparadas --resume results\campana_final --reintentar-python --python-max-segundos 1800
```

Si una repeticion individual necesita mas que el presupuesto, aumentar el limite;
no se puede reanudar a mitad de una convolucion. Una etapa fallida por error de
validacion requiere diagnostico; no se oculta como limitacion de tiempo.

Una terminacion abrupta del sistema puede impedir el ultimo registro de estado.
Los CSV completos escritos se conservan; el siguiente --resume trata la etapa
`en_curso` como interrumpida. No desconectar el equipo durante la escritura.

## 4. Resultados y alcance de medicion

| Archivo | Contenido |
|---|---|
| experimentos.json | Plan, hashes, entorno, sesiones, comandos e intentos por etapa. |
| mediciones/*/*_raw.csv | Repeticiones originales C++/CUDA C. |
| *_python_raw.csv / *_pycuda_raw.csv | Repeticiones Python/PyCUDA, errores y estado. |
| *_python_estado.json | Piloto, estimaciones, calentamiento, interrupcion y sesiones. |
| *_pycuda_preparacion.json | Inicializacion/contexto y compilacion/carga, separados. |
| *_pycuda_reservas.json | Reservas previas, excluidas del total GPU. |
| *_metadata.txt | Entorno C++, contexto runtime, reservas y aceleracion C++/CUDA C. |
| build/compilacion.json | Tiempo de configuracion y build C++/CUDA anticipado. |
| mediciones.csv | Filas disponibles; distingue estado de fila y de configuracion. |
| resumen.csv | Medianas, Q1, Q3, IQR y aceleraciones SOLO de series completas. |
| estado_configuraciones.csv | Estado, numero de repeticiones, limite y motivo. |
| cobertura.json | Pendientes y uso de alternativa documentada de Python. |
| figuras/ | Tiempos, aceleraciones, transferencias y comparativos visuales. |
| *.f32 | Referencias float32 con signo; conservadas para reanudar/validar. |

Dejar al menos **3 GB libres** para una campana y margen adicional para intentos.
Las referencias sin compresion y las imagenes ocupan bastante mas que el ZIP.

CPU: solo convolucion. C++ usa steady_clock y Python perf_counter; las salidas se
reservan y los coeficientes se preparan antes del cronometro. Validacion y archivos
quedan fuera. Un calentamiento completo precede a las cinco repeticiones.

GPU: kernel con eventos CUDA sincronizados. H2D y D2H con reloj host, y total desde
el inicio de H2D hasta terminar D2H. El total incluye lanzamiento, instrumentacion y
sincronizacion, pero no reservas, contexto, compilacion, validacion ni archivos.
Host pageable en ambas versiones GPU, sin solapamiento. Por ello total puede ser
mayor que H2D + kernel + D2H. Compilacion de CUDA C es anticipada (build); PyCUDA
registra compilacion/carga de SourceModule, con posible uso de su cache. Cualquier
carga diferida del runtime C++ se amortiza mediante calentamiento; no se atribuye
un tiempo ficticio aislado a ella.

Las ejecuciones de C++/CUDA C alternan orden por repeticion; las implementaciones
Python se ejecutan despues, en procesos distintos. Comparar medianas, no restar
repeticiones como si fueran simultaneas. Frecuencias, temperatura, WDDM y carga del
laboratorio pueden afectar diferencias pequenas. Conservar valores extremos.

Para regenerar tablas y figuras sin medir:

```bat
python scripts\graficar.py results\campana_final
```

## 5. Aceleracion y eficiencia

El calculo principal esta en **scripts/graficar.py, funcion calcular_aceleraciones**.
Usa cocientes de medianas completas y validadas, con la misma entrada/filtro/mascara:

- CUDA C kernel: mediana(C++) / mediana(kernel CUDA C).
- CUDA C total: mediana(C++) / mediana(total CUDA C).
- PyCUDA kernel: mediana(Python) / mediana(kernel PyCUDA).
- PyCUDA total: mediana(Python) / mediana(total PyCUDA).

Columnas: `referencia_aceleracion`, `aceleracion_kernel`, `aceleracion_total`.
Tambien se calcula CUDA C / PyCUDA para kernel y total; un cociente mayor que uno
significa que PyCUDA tuvo menor tiempo en esa comparacion. Las bases de las dos
aceleraciones son distintas: un mayor speedup de PyCUDA no implica que su kernel
sea mejor que CUDA C. Para eso comparar tiempos GPU directamente.

En **src/main.cpp** tambien se calcula C++/CUDA C, despues de las repeticiones, para
el archivo *_metadata.txt. No es un segundo calculo de Python/PyCUDA.

La eficiencia paralela clasica E(p)=S(p)/p requiere p unidades comparables y una
referencia serial apropiada. En esta comparacion CPU-GPU heterogenea no se identifica
ese p de manera valida. **No dividir por 128/256/512 hilos por bloque, por el total de
hilos lanzados ni por los CUDA cores.** Cambiar el bloque no cambia el numero de GPU
ni de SM disponibles. El enunciado no exige esa eficiencia. El campo queda vacio con
el motivo indicado. Tampoco se infiere ocupacion, eficiencia energetica o uso de GPU
solo a partir de tiempos.

Se reportan proporciones descriptivas (H2D+D2H)/total y kernel/total; NO se llaman
eficiencia paralela. Ver `ACELERACION_Y_EFICIENCIA.md`.

## 6. Filtros y validacion

C++ secuencial en src/cpu; Python por clases en src/python; CUDA C en src/gpu;
PyCUDA por clases/kernels en src/pycuda. Los scripts coordinan lectura y medicion.

- Media: k=3,9,21; coeficiente float32 1/(k*k), divisor fijo en bordes.
- Nitidez: [0,-1,0; -1,5,-1; 0,-1,0].
- Laplaciano: [0,1,0; 1,-4,1; 0,1,0].
- Un pixel por hilo GPU; bloques/hilos exclusivamente X, vecinos en memoria global.
- indice=blockIdx.x*blockDim.x+threadIdx.x; fila=indice/ancho, columna=indice%ancho.
- Entrada/salida separadas; limite indice<N y vecinos dentro de imagen.
- Datos y acumulacion float32, arrays NumPy contiguos C; anchos/altos/k np.int32.
- FMA deshabilitada en ambos kernels CUDA, sin fast-math.
- Errores maximo y medio calculados en float64 ANTES de abs/clip/redondeo visual.
- Tolerancia absoluta 0.001 intensidad: umbral practico de concordancia float32 muy
  inferior a un nivel de gris, no cota universal respecto a aritmetica exacta.
  Orden de acumulacion equivalente y FMA deshabilitada reducen discrepancias. Con
  entradas enteras nitidez/laplaciano tienen operaciones exactamente representables
  en este rango. No aumentar la tolerancia para ocultar errores; NaN/Inf invalidan.
- Visual: media/nitidez clip 0..255 y redondeo; laplaciano abs, clip y redondeo.

Matriz manual [1,2,3;4,5,6;7,8,9]:
media [12,21,16;27,45,33;24,39,28]/9;
nitidez [-1,1,7;7,5,13;23,19,31];
laplaciano [2,1,-4;-3,0,-7;-16,-11,-22].
Se prueban 1x1, 3x3, 17x19 (323 pixeles, no multiplo de 128/256/512), impulsos
con negativos y valores >255, y rectangulos adicionales C++/Python.

## 7. Evidencia anterior, versionado y pendientes reales

`evidencia_original/` no se modifica: contiene los CSV, imagenes, referencias .f32,
manifest original, codigo previo y registro de pruebas reales del laboratorio.
Su manifest seguia en curso; se conserva tal cual por trazabilidad.

`revision_resultados_originales/` contiene la consolidacion corregida de ESA evidencia:
cuatro configuraciones de cuatro versiones y C++/CUDA C completos de la quinta.
La unica repeticion Python de la quinta se conserva, sin speedup ni estadistica de
configuracion completa. Estos tiempos no pertenecen a la version nueva de Python,
que ahora prepara memoria/coeficientes fuera del cronometro.

Para reconstruir esa revision sin cambiar originales:

```bat
python scripts\graficar.py evidencia_original\results\experimentos_20261110_152824 --out revision_resultados_originales
```

El snapshot original identifica archivos y hashes, pero no puede recuperar el commit
original de practica 01: el laboratorio devolvio `not a git repository`. El archivo
`VERSION_CODIGO.json` identifica el commit local del codigo corregido; el bundle
incluye ese historial para poder publicarlo sin perder su identidad:

```bat
git clone codigo_practica02.bundle repositorio_practica02
```

Ese repositorio de codigo referencia las entradas del paquete y sus hashes. Copiar
los datos del ZIP a su carpeta data si se ejecuta desde el clone. Agregar el remoto
GitHub propio y publicar ese commit; registrar tambien el commit real de practica 01.
No se inventa un enlace GitHub ni autoria/origen de las imagenes. Completar los datos
que solo los integrantes conocen en data/README.md y conservar entradas.json.
`requirements-laboratorio-20261110.txt` guarda las versiones exactas anteriores;
la nueva campana registra de nuevo su entorno y pip freeze.

La correccion fue verificada sin GPU: compilacion C++, pruebas CPU, equivalencia
C++/Python y regresiones de medicion/reanudacion/estadisticas. Los kernels conservan
las operaciones originales, pero la compilacion y ejecucion CUDA del paquete nuevo
deben verificarse en el laboratorio con compilar_windows.bat. Los logs previos son
evidencia de la version anterior, no certificacion del paquete corregido.

## Referencias tecnicas

- Enunciado adjunto de practica 02, especialmente viabilidad Python y medicion.
- https://documen.tician.de/pycuda/driver.html
- https://documen.tician.de/pycuda/tutorial.html
- https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html
