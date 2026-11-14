# Aceleracion y eficiencia en esta practica

## Donde se calcula la aceleracion

En `scripts/graficar.py`, funcion `calcular_aceleraciones(medianas)`. Se llama desde
`resumir` despues de comprobar las cinco repeticiones y calcular sus medianas.

| Comparacion | Numerador | Denominador | Columna |
|---|---|---|---|
| CUDA C: computo | Mediana C++ | Mediana kernel CUDA C | aceleracion_kernel |
| CUDA C: con transferencias | Mediana C++ | Mediana total CUDA C | aceleracion_total |
| PyCUDA: computo | Mediana Python | Mediana kernel PyCUDA | aceleracion_kernel |
| PyCUDA: con transferencias | Mediana Python | Mediana total PyCUDA | aceleracion_total |

`referencia_aceleracion` identifica CPU (C++) o PYTHON. No se calculan cocientes
cuando falta una serie completa. Todos los tiempos del cociente estan en milisegundos.
Las aceleraciones son adimensionales y se expresan como x, no como porcentaje.
Son cocientes de medianas, no medianas de cocientes de repeticiones no emparejadas.

En `src/main.cpp` se calcula adicionalmente C++/CUDA C para *_metadata.txt. La
consolidacion recalcula desde CSV originales, sin usar texto redondeado de consola.

## Ejemplo REAL de tus registros anteriores

Media 3x3, 4000x4000, 256 hilos, cinco repeticiones:

- Mediana C++: 116.5362 ms.
- Mediana Python: 34437.903 ms.
- Mediana total CUDA C: 16.8723 ms.
- Mediana total PyCUDA: 17.0781 ms.
- CUDA C respecto a C++: aproximadamente 6.91x.
- PyCUDA respecto a Python: aproximadamente 2016.49x.

El numero de PyCUDA es mayor principalmente por la referencia Python mucho mas
lenta; los tiempos GPU totales son cercanos. No concluir que PyCUDA es miles de
veces mas rapido que CUDA C. Estos datos son del codigo previo, conservados como
evidencia; no son mediciones de la version corregida.

## Aplica la eficiencia E = S/p?

La definicion clasica de eficiencia paralela es E(p)=T1/(p*Tp)=S(p)/p. Describe el
aprovechamiento relativo de p unidades de procesamiento bajo un modelo y una
referencia serial comparables.

En este experimento se compara una CPU secuencial con una GPU, y ademas hay
referencias escritas en lenguajes diferentes. Los CUDA cores y los nucleos CPU no
son unidades equivalentes. Los hilos CUDA son trabajo logico que el hardware
planifica; no son procesadores independientes disponibles simultaneamente.

Por eso **no se reporta E=S/256 ni S/numero_de_CUDA_cores**. Los 128, 256 y 512
son hilos por bloque; cambiar ese parametro no cambia el numero fisico de SM/GPU.
Una configuracion de bloque puede afectar ocupacion y rendimiento, pero este
benchmark por si solo no mide esas metricas.

La eficiencia paralela clasica no es una metrica interpretable para esta comparacion
tal como esta disenada y no es exigida por el enunciado. Si se quisiera estudiarla,
habria que definir otro experimento con recursos comparables y un p controlado,
por ejemplo escalabilidad sobre varios numeros de nucleos CPU o varias GPU con
referencias apropiadas. No se requiere ampliar la practica de esa manera.

## Que si se reporta

- Tiempos y variabilidad (mediana, Q1, Q3, IQR).
- Aceleraciones con la referencia correcta.
- Comparacion directa CUDA C/PyCUDA: razon de tiempos GPU, con y sin transferencias.
- Proporcion de transferencias: 100*(H2D+D2H)/total, mediana de proporciones por repeticion.
- Fraccion temporal del kernel: 100*kernel/total, mediana por repeticion.

Las dos ultimas son distribuciones del tiempo, no eficiencia paralela ni ocupacion.
Tampoco puede calcularse eficiencia energetica sin medir consumo de energia.

Referencias:
https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html
https://documen.tician.de/pycuda/driver.html
