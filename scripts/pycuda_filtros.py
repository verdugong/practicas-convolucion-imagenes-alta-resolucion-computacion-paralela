"""Entrada de medicion para los filtros implementados en src/pycuda."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from comun import guardar_json, ahora, interrupciones
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pycuda.filtros_gpu import (  # noqa: E402
    CODIGO_CUDA,
    FiltroPyCUDA,
    FiltroMediaPyCUDA,
    FiltroNitidezPyCUDA,
    FiltroLaplacianoPyCUDA,
    crear_modulo,
)

TOLERANCIA = 0.001


def cargar_imagen(ruta):
    """Cargar la misma entrada PGM P5 que utiliza C++."""
    with Path(ruta).open("rb") as f:
        if f.read(2) != b"P5":
            raise ValueError("Se requiere PGM binario P5.")
    with Image.open(ruta) as im:
        if im.mode != "L":
            raise ValueError("Se requiere gris de 8 bits.")
        return np.array(im, dtype=np.float32, order="C")


def cargar_referencia(ruta, forma):
    """Leer una referencia float32 sin perdida."""
    if Path(ruta).stat().st_size != int(np.prod(forma)) * 4:
        raise ValueError(f"Referencia incompleta o dimensiones incorrectas: {ruta}")
    return np.fromfile(ruta, dtype="<f4").reshape(forma)


def errores(referencia, resultado):
    """Calcular errores en float64, antes de visualizar."""
    if referencia.shape != resultado.shape or not referencia.size:
        raise ValueError("Dimensiones incompatibles al validar.")
    if not np.isfinite(referencia).all() or not np.isfinite(resultado).all():
        return float("inf"), float("inf")
    ref, out = referencia.ravel(), resultado.ravel()
    maximo, suma = 0.0, 0.0
    for inicio in range(0, ref.size, 1_000_000):
        diferencia = np.abs(
            ref[inicio : inicio + 1_000_000].astype(np.float64)
            - out[inicio : inicio + 1_000_000].astype(np.float64)
        )
        maximo = max(maximo, float(diferencia.max()))
        suma += float(diferencia.sum())
    return maximo, suma / ref.size


def guardar_visual(ruta, resultado, filtro):
    valores = np.abs(resultado) if filtro == "laplaciano" else resultado
    pixeles = np.floor(np.clip(valores, 0, 255) + 0.5).astype(np.uint8)
    Image.fromarray(pixeles).save(ruta)


def medir(modulo, entrada, cpu, cuda_c, filtro, k, hilos, reps, prefijo):
    """Medir y validar PyCUDA frente a CPU y CUDA C."""
    if not 5 <= reps <= 1000:
        raise ValueError("Se requieren entre 5 y 1000 repeticiones.")
    prefijo = Path(prefijo)
    archivo_csv = Path(str(prefijo) + "_pycuda_raw.csv")
    imagen = Path(str(prefijo) + "_pycuda.pgm")
    if archivo_csv.exists() or imagen.exists():
        raise FileExistsError("El prefijo PyCUDA ya existe; use otro.")
    prefijo.parent.mkdir(parents=True, exist_ok=True)
    reserva_inicio = perf_counter()
    gpu = FiltroPyCUDA(modulo, entrada, filtro, k, hilos)
    reserva_ms = (perf_counter() - reserva_inicio) * 1000
    guardar_json(str(prefijo) + "_pycuda_reservas.json", {"reserva_ms": reserva_ms,
        "incluida_en_total_gpu": False, "memoria_host": "pageable", "unidad": "ms"})
    columnas = [
        "implementacion", "filtro", "ancho", "alto", "mascara", "hilos", "bloques",
        "repeticion", "cpu_ms", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms",
        "error_max", "error_medio", "tolerancia", "validado", "unidad",
        "error_cuda_max", "error_cuda_medio", "validado_cuda", "estado",
    ]
    try:
        gpu.ejecutar()
        if max(errores(cpu, gpu.salida)[0], errores(cuda_c, gpu.salida)[0]) > TOLERANCIA:
            raise ValueError("PyCUDA difiere de CPU/CUDA C durante calentamiento.")
        with archivo_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columnas)
            writer.writeheader()
            for repeticion in range(1, reps + 1):
                tiempos = gpu.ejecutar()
                emax, emedio = errores(cpu, gpu.salida)
                cmax, cmedio = errores(cuda_c, gpu.salida)
                valido, valido_cuda = emax <= TOLERANCIA, cmax <= TOLERANCIA
                writer.writerow(dict(
                    implementacion="PYCUDA", filtro=filtro, ancho=entrada.shape[1],
                    alto=entrada.shape[0], mascara=k, hilos=hilos, bloques=gpu.bloques,
                    repeticion=repeticion, cpu_ms="", **tiempos, error_max=emax,
                    error_medio=emedio, tolerancia=TOLERANCIA,
                    validado=str(valido).lower(), unidad="ms", error_cuda_max=cmax,
                    error_cuda_medio=cmedio, validado_cuda=str(valido_cuda).lower(),
                    estado="completada" if valido and valido_cuda else "fallida",
                ))
                f.flush()
                if not valido or not valido_cuda:
                    raise ValueError("PyCUDA difiere: se conserva el CSV del fallo.")
                print(
                    f"PyCUDA {repeticion}/{reps}: kernel={tiempos['kernel_ms']:.6f} ms, "
                    f"total={tiempos['total_ms']:.6f} ms, error CPU={emax}, CUDA C={cmax}",
                    flush=True,
                )
        guardar_visual(imagen, gpu.salida, filtro)
    finally:
        gpu.cerrar()
    print("VALIDACION CPU-CUDA C-PYCUDA CORRECTA", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="Verificar PyCUDA en GPU 0")
    p.add_argument("--input", type=Path)
    p.add_argument("--filter", choices=["media", "nitidez", "laplaciano"], default="media")
    p.add_argument("--mask", type=int, default=3)
    p.add_argument("--threads", type=int, default=256)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--reference-prefix", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if not a.check and (a.input is None or a.reference_prefix is None or a.output is None):
        p.error("Usar --input, --reference-prefix y --output, o --check.")
    import pycuda
    import pycuda.driver as cuda

    interrupciones()
    t0 = perf_counter()
    cuda.init()
    contexto = cuda.Device(0).make_context()
    contexto_ms = (perf_counter() - t0) * 1000
    try:
        t0 = perf_counter()
        modulo = crear_modulo()
        compilacion_carga_ms = (perf_counter() - t0) * 1000
        preparacion = {"inicio_utc": ahora(), "contexto_inicializacion_ms": contexto_ms,
            "compilacion_carga_ms": compilacion_carga_ms, "unidad": "ms",
            "cache_compilador": "habilitada por defecto; SourceModule puede reutilizarla",
            "reservas_en_total_gpu": False, "incluida_en_tiempos_kernel": False}
        if not a.check:
            guardar_json(str(a.output) + "_pycuda_preparacion.json", preparacion)
        if a.check:
            entrada = np.array([[9]], dtype=np.float32)
            gpu = FiltroPyCUDA(modulo, entrada, "media", 3, 128)
            try:
                gpu.ejecutar()
                if errores(np.array([[1]], dtype=np.float32), gpu.salida)[0] > TOLERANCIA:
                    raise ValueError("Fallo prueba inicial PyCUDA.")
            finally:
                gpu.cerrar()
            print(json.dumps({
                "pycuda": pycuda.VERSION_TEXT, "numpy": np.__version__,
                "gpu": cuda.Device(0).name(), "pci": cuda.Device(0).pci_bus_id(),
                "capability": cuda.Device(0).compute_capability(),
                "driver": cuda.get_driver_version(), "toolkit_pycuda": cuda.get_version(),
                "opciones_nvcc": ["-std=c++17", "--fmad=false", "-O3"],
            }))
        else:
            entrada = cargar_imagen(a.input)
            cpu = cargar_referencia(str(a.reference_prefix) + "_cpu.f32", entrada.shape)
            cuda_c = cargar_referencia(str(a.reference_prefix) + "_cuda.f32", entrada.shape)
            medir(modulo, entrada, cpu, cuda_c, a.filter, a.mask, a.threads, a.reps, a.output)
    finally:
        modulo = None
        contexto.pop()
        contexto.detach()


if __name__ == "__main__":
    main()
