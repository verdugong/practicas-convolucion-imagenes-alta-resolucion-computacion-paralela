"""Validar los tres filtros secuenciales implementados en Python."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.python.filtros_python import (  # noqa: E402
    FiltroMediaCPU,
    FiltroNitidezCPU,
    FiltroLaplacianoCPU,
)


def exigir(condicion, mensaje):
    if not condicion:
        raise AssertionError(mensaje)


def main():
    entrada = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
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
    filtros = {
        "media": FiltroMediaCPU(3),
        "nitidez": FiltroNitidezCPU(),
        "laplaciano": FiltroLaplacianoCPU(),
    }
    for nombre, filtro in filtros.items():
        salida = filtro.ejecutar(entrada)
        exigir(
            np.allclose(salida, esperados[nombre], atol=1e-6, rtol=0),
            f"Fallo matriz {nombre}.",
        )
        exigir(salida.dtype == np.float32, f"Tipo incorrecto en {nombre}.")
    impulso = np.zeros((3, 3), dtype=np.float32)
    impulso[1, 1] = 255
    exigir(
        FiltroNitidezCPU().ejecutar(impulso)[1, 1] == 1275
        and FiltroNitidezCPU().ejecutar(impulso)[0, 1] == -255,
        "Nitidez se recorto antes de validar.",
    )
    laplaciano = FiltroLaplacianoCPU().ejecutar(impulso)
    exigir(laplaciano[1, 1] == -1020 and laplaciano[0, 1] == 255,
           "Laplaciano perdio signo.")
    for k in (3, 9, 21):
        salida = FiltroMediaCPU(k).ejecutar(np.array([[9]], dtype=np.float32))
        exigir(salida[0, 0] == np.float32(9) / np.float32(k * k),
               "Fallo borde 1x1.")
    print("PRUEBAS CPU PYTHON CORRECTAS: matriz manual, bordes e impulso.")


if __name__ == "__main__":
    main()
