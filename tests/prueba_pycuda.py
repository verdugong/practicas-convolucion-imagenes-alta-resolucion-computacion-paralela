"""Comparar PyCUDA con referencias float32 de CPU y CUDA C en casos pequenos.

La matriz manual se comprueba directamente. Para 1x1, 3x3 y 17x19, se ejecutan
las cinco combinaciones filtro/mascara y los tres tamanos de bloque (45 casos).
No se utiliza NumPy como referencia de rendimiento de CPU.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pycuda_filtros import (
    FiltroPyCUDA,
    crear_modulo,
    errores,
    cargar_referencia,
    TOLERANCIA,
)


from src.python.filtros_python import crear_filtro_cpu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True, type=Path)
    args = p.parse_args()
    import pycuda.driver as cuda

    cuda.init()
    contexto = cuda.Device(0).make_context()
    modulo = None
    try:
        modulo = crear_modulo()
        manual = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        esperados = {
            "media": np.array(
                [[12, 21, 16], [27, 45, 33], [24, 39, 28]], dtype=np.float32
            )
            / np.float32(9),
            "nitidez": np.array(
                [[-1, 1, 7], [7, 5, 13], [23, 19, 31]], dtype=np.float32
            ),
            "laplaciano": np.array(
                [[2, 1, -4], [-3, 0, -7], [-16, -11, -22]], dtype=np.float32
            ),
        }
        y, x = np.indices((19, 17))
        demo = ((x * 17 + y * 29) % 256).astype(np.float32)
        cantidad = 0
        with tempfile.TemporaryDirectory(prefix="practica_pycuda_") as carpeta:
            root = Path(carpeta)
            for indice, entrada in enumerate(
                [np.array([[9]], dtype=np.float32), manual, demo]
            ):
                ruta = root / f"entrada_{indice}.pgm"
                Image.fromarray(entrada.astype(np.uint8)).save(ruta)
                for filtro, k in [
                    ("media", 3),
                    ("media", 9),
                    ("media", 21),
                    ("nitidez", 3),
                    ("laplaciano", 3),
                ]:
                    prefijo = root / f"caso_{indice}_{filtro}_{k}"
                    # El mismo ejecutable produce referencia CPU y resultado CUDA C sin recorte.
                    proceso = subprocess.run(
                        [
                            str(args.exe.resolve()),
                            "--input",
                            str(ruta),
                            "--filter",
                            filtro,
                            "--mask",
                            str(k),
                            "--reps",
                            "5",
                            "--output",
                            str(prefijo),
                            "--export-float",
                        ],
                        capture_output=True,
                        text=True,
                        errors="replace",
                    )
                    if proceso.returncode:
                        raise RuntimeError(proceso.stdout + proceso.stderr)
                    cpu = cargar_referencia(str(prefijo) + "_cpu.f32", entrada.shape)
                    cuda_c = cargar_referencia(
                        str(prefijo) + "_cuda.f32", entrada.shape
                    )
                    python = crear_filtro_cpu(filtro, k).ejecutar(entrada)
                    if errores(cpu, python)[0] > TOLERANCIA:
                        raise AssertionError("Python difiere de C++ en prueba pequena/de limites.")
                    for hilos in (128, 256, 512):
                        gpu = FiltroPyCUDA(modulo, entrada, filtro, k, hilos)
                        try:
                            gpu.ejecutar()
                            ecpu = errores(cpu, gpu.salida)[0]
                            ecuda = errores(cuda_c, gpu.salida)[0]
                            if max(ecpu, ecuda) > TOLERANCIA:
                                raise AssertionError(
                                    f"{filtro}, k{k}, t{hilos}: CPU={ecpu}, CUDA C={ecuda}"
                                )
                            if (
                                indice == 1
                                and k == 3
                                and errores(esperados[filtro], gpu.salida)[0]
                                > TOLERANCIA
                            ):
                                raise AssertionError(
                                    "Fallo de la matriz manual PyCUDA."
                                )
                        finally:
                            gpu.cerrar()
                        cantidad += 1
            # Un impulso comprueba explicitamente que se conserva signo y valores >255.
            impulso = np.zeros((3, 3), dtype=np.float32)
            impulso[1, 1] = 255
            for filtro, centro, vecino in [
                ("nitidez", 1275, -255),
                ("laplaciano", -1020, 255),
            ]:
                gpu = FiltroPyCUDA(modulo, impulso, filtro, 3, 128)
                try:
                    gpu.ejecutar()
                    if gpu.salida[1, 1] != centro or gpu.salida[0, 1] != vecino:
                        raise AssertionError(
                            "Se perdio el signo o se saturo antes de validar."
                        )
                finally:
                    gpu.cerrar()
        print(
            f"PRUEBAS PYCUDA CORRECTAS: {cantidad} casos C++/CUDA C/Python/PyCUDA, matriz manual e impulso."
        )
    finally:
        modulo = None
        contexto.pop()
        contexto.detach()


if __name__ == "__main__":
    main()
