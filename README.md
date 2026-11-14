# Practica 2: convolucion de imagenes con CPU, CUDA C, Python y PyCUDA

Esta practica compara cuatro implementaciones de los mismos filtros de
convolucion sobre imagenes de alta resolucion:

- CPU en C++.
- GPU con CUDA C.
- CPU secuencial en Python.
- GPU con PyCUDA.

El objetivo es medir tiempos de ejecucion, transferencias y aceleraciones sin
confundir la preparacion de datos, la compilacion o la validacion con el tiempo
del algoritmo.

## 1. Estructura

| Ruta | Funcion |
|---|---|
| `CMakeLists.txt` | Compilacion C++17/CUDA y registro de pruebas CTest. |
| `requirements.txt` | NumPy, Pillow, matplotlib, psutil y PyCUDA. |
| `compilar_windows.bat` | Comprueba el entorno, compila y ejecuta las pruebas. |
| `ejecutar_windows.bat` | Ejecuta una campana completa de medicion. |
| `include/` | Declaraciones de imagenes, filtros, validacion y kernels CUDA. |
| `src/main.cpp` | Ejecutable C++: filtros, mediciones, validacion y CSV. |
| `src/cpu/` | Implementaciones secuenciales en C++. |
| `src/gpu/` | Kernels y mediciones de CUDA C. |
| `src/python/` | Implementaciones secuenciales en Python. |
| `src/pycuda/` | Kernels y clases para ejecutar los filtros con PyCUDA. |
| `scripts/compilar.py` | Configura y compila el proyecto C++/CUDA. |
| `scripts/experimentos.py` | Ejecuta las configuraciones y guarda el estado. |
| `scripts/graficar.py` | Consolida CSV, calcula aceleraciones y genera figuras. |
| `scripts/convertir_entradas.py` | Prepara las imagenes en formato PGM. |
| `scripts/pycuda_filtros.py` | Comprueba o ejecuta directamente los filtros PyCUDA. |
| `scripts/comparar_imagenes.py` | Genera comparativos visuales de las salidas. |
| `tests/` | Pruebas CPU, GPU, pipeline y equivalencia con PyCUDA. |
| `data/originales/` | Imagenes originales; no se publican automaticamente. |
| `data/preparadas/` | Entradas PGM y metadatos de preparacion. |
| `results/` | CSV, metadatos y figuras de los experimentos hechos. |
| `defensa_gpu/` | Comprobacion auxiliar de un caso individual en GPU para la exposición. |
| `build/` | Archivos de compilacion local; se excluye de Git. |

## 2. Compilar y comprobar

Entorno previsto: Windows x64, Visual Studio 2022 C++, CUDA Toolkit, CMake
>=3.24, Python de 64 bits >=3.10 y una GPU NVIDIA. La configuracion utilizada
para las mediciones principales es una RTX 3070, arquitectura CUDA 86.

Abrir **x64 Native Tools Command Prompt for VS 2022**, entrar en la carpeta con
`CMakeLists.txt` e instalar las dependencias:

```bat
python -m pip install -r requirements.txt
```

Para compilar y ejecutar todas las pruebas:

```bat
compilar_windows.bat
```

El script comprueba `cl`, `nvcc`, Python y PyCUDA, compila el ejecutable
`build\practica1.exe` y ejecuta CTest. Con CUDA y Python disponibles se validan:

1. Referencia CPU en C++.
2. Referencia CPU en Python.
3. Regresiones del pipeline.
4. Equivalencia CPU-GPU con CUDA C.
5. Equivalencia CPU-GPU con PyCUDA.

El registro queda en `verificacion_gpu_laboratorio.txt`. Si solo se necesita
comprobar la referencia CPU en un equipo sin CUDA:

```bat
python scripts\compilar.py --cpu-only
ctest --test-dir build -C Release --output-on-failure
```

Esta comprobacion CPU no sustituye la validacion final de CUDA y PyCUDA.

## 3. Preparar las entradas

La practica utiliza una misma imagen y encuadre en tres resoluciones:

- `imagen_2000.pgm`: 2000 x 2000 pixeles.
- `imagen_4000.pgm`: 4000 x 4000 pixeles.
- `imagen_6000.pgm`: 6000 x 6000 pixeles.

Las entradas PGM preparadas ya se encuentran en `data/preparadas/`. No ejecutar
de nuevo el conversor sobre esa misma carpeta porque no sobrescribe resultados
existentes. Para preparar otra copia:

```bat
python scripts\convertir_entradas.py ^
  --img2000 "data\originales\foto_2000.jpg" ^
  --img4000 "data\originales\foto_4000.jpg" ^
  --img6000 "data\originales\foto_6000.jpg" ^
  --out data\preparadas_nueva
```

El conversor verifica dimensiones, corrige orientacion EXIF, convierte a escala
de grises de 8 bits, genera PGM binario P5 y registra hashes SHA-256 en
`entradas.json`. No redimensiona ni recorta durante la conversion. La
procedencia de la imagen se documenta en `data/README.md`.

## 4. Prueba rapida y experimentos

Para comprobar directamente PyCUDA en la GPU:

```bat
python scripts\pycuda_filtros.py --check
```

Para ejecutar una campana reproducible con cinco repeticiones:

```bat
python scripts\experimentos.py ^
  --exe build\practica1.exe ^
  --data data\preparadas ^
  --out results\campana_final ^
  --reps 5 ^
  --python-max-segundos 120
```

Tambien se puede usar:

```bat
ejecutar_windows.bat
```

La campana ejecuta CUDA C y PyCUDA para las configuraciones principales,
configuraciones adicionales con bloques de 512 hilos y Python secuencial en
los casos que el tiempo de laboratorio permite.

| Experimento | Variable | Condiciones principales |
|---|---|---|
| Filtros | Media, nitidez y laplaciano | 4000x4000, mascara 3, 256 hilos |
| Resolucion | 2000, 4000 y 6000 pixeles | Media, mascara 9, 256 hilos |
| Mascara | 3, 9 y 21 | Media, 4000x4000, 256 hilos |
| Hilos | 128, 256 y 512 | Media, 4000x4000, mascara 9 |

Las configuraciones con Python pueden tardar considerablemente mas. El limite
predeterminado de 120 segundos se aplica por proceso Python grande. Para no
limitar estas mediciones:

```bat
ejecutar_windows.bat --python-max-segundos 0
```

Para detener y reanudar una campana, usar directamente:

```bat
python scripts\experimentos.py ^
  --exe build\practica1.exe ^
  --data data\preparadas ^
  --resume results\campana_final
```

Una configuracion que no termina queda marcada como `no_completada` y no se usa
para calcular aceleraciones. No se deben completar resultados manualmente.

## 5. Salidas

Cada campana genera una carpeta dentro de `results` con archivos como estos:

- `experimentos.json`: plan, entradas, hashes, equipo y sesiones.
- `mediciones.csv`: filas disponibles de todas las implementaciones.
- `resumen.csv`: medianas, cuartiles, IQR y aceleraciones validas.
- `estado_configuraciones.csv`: repeticiones, limites y motivos de cada caso.
- `*_raw.csv`: mediciones originales de CPU, CUDA C, Python o PyCUDA.
- `*_metadata.txt`: entorno, configuracion y mediciones de CUDA C.
- `*_python_estado.json`: piloto, calentamiento, limite e interrupciones.
- `*_pycuda_preparacion.json`: contexto y compilacion de kernels PyCUDA.
- `*_pycuda_reservas.json`: reservas previas al cronometro.
- `figuras/`: tiempos, aceleraciones, transferencias y comparativos visuales.

Para regenerar tablas y figuras sin ejecutar otra campana:

```bat
python scripts\graficar.py results\campana_final
```

Las referencias `.f32`, imagenes intermedias y salidas PGM pueden ocupar varios
gigabytes. Se conservan para validar o reanudar localmente, pero se excluyen
segun las reglas de `.gitignore` cuando no son necesarias para la entrega.

## 6. Algoritmos y validacion

Las cuatro implementaciones aplican los mismos filtros:

| Filtro | Coeficientes | Efecto |
|---|---|---|
| Media k x k | Todos `1/(k*k)` | Suaviza y reduce detalle. |
| Nitidez 3 x 3 | `[0,-1,0; -1,5,-1; 0,-1,0]` | Realza cambios locales. |
| Laplaciano 3 x 3 | `[0,1,0; 1,-4,1; 0,1,0]` | Detecta cambios de intensidad. |

En las versiones GPU se utiliza un hilo por pixel, bloques de hilos en X,
memoria global y arreglos separados de entrada y salida. Los limites de la
imagen se tratan como ceros y la media mantiene el divisor completo en los
bordes. La entrada original se usa como origen independiente para cada filtro.

PyCUDA construye los kernels CUDA desde cadenas de codigo mediante
`SourceModule`. La compilacion del kernel y la reserva de memoria se realizan
antes de medir la ejecucion principal. Los kernels usan `float32`, desactivan
FMA y no utilizan `fast-math`, para facilitar la comparacion con CUDA C y Python.

Las pruebas comparan las salidas antes de visualizarlas:

- Error maximo y error medio en `float64`.
- Tolerancia absoluta de `0.001` unidades de intensidad.
- Rechazo de valores NaN o infinitos.
- Casos 1x1, 3x3, 17x19, impulsos y resoluciones grandes.

La visualizacion redondea y satura media y nitidez a `[0,255]`. El laplaciano
visualiza el valor absoluto saturado; las salidas float originales conservan el
signo y se validan antes de esa transformacion.

## 7. Metodologia de medicion

Cada configuracion realiza un calentamiento y despues cinco repeticiones, salvo
que se indique otro valor con `--reps`. Las reservas, la compilacion, la
validacion y la escritura de archivos quedan fuera del tiempo del algoritmo.

- CPU C++: `steady_clock`.
- CPU Python: `perf_counter`.
- Kernel CUDA C y PyCUDA: eventos CUDA.
- Transferencias H2D y D2H: reloj del host.
- GPU total: desde H2D hasta finalizar D2H.

El tiempo total GPU incluye lanzamiento, instrumentacion y sincronizacion, por lo
que puede ser mayor que la suma aislada de H2D, kernel y D2H. Las ejecuciones
Python se realizan en procesos separados de las ejecuciones CUDA C y PyCUDA.

Las aceleraciones se calculan con medianas de series completas:

- CUDA C: CPU C++ / CUDA C.
- PyCUDA: Python / PyCUDA.
- Comparacion directa adicional: CUDA C / PyCUDA.

No se divide por 128, 256 o 512 para afirmar una eficiencia paralela: esos son
tamanos de bloque, no el numero de GPU ni de unidades de ejecucion comparables.
Las proporciones de transferencias y kernel se reportan como medidas descriptivas,
no como eficiencia paralela clasica.

## 8. GitHub y entrega

Versionar el codigo fuente, README, pruebas, configuraciones, CSV resumidos,
metadatos necesarios y figuras finales. Mantener fuera del repositorio:

- `build/` y ejecutables.
- `__pycache__/` y archivos `.pyc`.
- Imagenes originales grandes si no son necesarias.
- Referencias `.f32` y PGM generadas dentro de `results`.
- Archivos temporales, bundles y logs locales.

Las conclusiones deben obtenerse de las mediciones reales del equipo. El paquete
incluye herramientas y evidencia de laboratorio, pero no sustituye el informe
academico ni la presentacion solicitados por el enunciado.

Antes de entregar:

```bat
git rev-parse HEAD
git status
```

Revisar tambien que no se hayan incluido credenciales, archivos generados o
resultados de otra maquina.

## 9. Verificacion y referencias

La referencia CPU, el pipeline y las comparaciones de resultados se pueden
comprobar sin modificar las mediciones:

```bat
compilar_windows.bat
ctest --test-dir build -C Release --output-on-failure
```

La validacion CUDA C y PyCUDA requiere una GPU NVIDIA, CUDA Toolkit, un
controlador compatible y PyCUDA instalado en el mismo Python utilizado por CTest.
Los tiempos dependen de la GPU, temperatura, frecuencia, controlador y carga del
equipo; no deben interpretarse como valores universales.

Referencias tecnicas:

- Enunciado de la practica 2.
- https://documen.tician.de/pycuda/driver.html
- https://documen.tician.de/pycuda/tutorial.html
- https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html
