"""Coordinador de filtros PyCUDA, equivalente a src/gpu/filtros_gpu.cu."""

from pathlib import Path
import numpy as np

from .base import FiltroPyCUDA
from .laplaciano_gpu import CODIGO as CODIGO_LAPLACIANO, FiltroLaplacianoPyCUDA
from .media_gpu import CODIGO as CODIGO_MEDIA, FiltroMediaPyCUDA
from .nitidez_gpu import CODIGO as CODIGO_NITIDEZ, FiltroNitidezPyCUDA

CODIGO_CUDA = "#include <stddef.h>\n" + CODIGO_MEDIA + CODIGO_NITIDEZ + CODIGO_LAPLACIANO


def crear_modulo():
    """Compilar los kernels antes de medir con un contexto CUDA activo."""
    from pycuda.compiler import SourceModule

    return SourceModule(CODIGO_CUDA, options=["-std=c++17", "--fmad=false", "-O3"])


def crear_filtro(modulo, entrada, filtro, k, hilos):
    """Crear la clase PyCUDA concreta solicitada."""
    clases = {
        "media": FiltroMediaPyCUDA,
        "nitidez": FiltroNitidezPyCUDA,
        "laplaciano": FiltroLaplacianoPyCUDA,
    }
    try:
        return clases[filtro](modulo, entrada, filtro, k, hilos)
    except KeyError as exc:
        raise ValueError("Filtro invalido.") from exc


__all__ = [
    "CODIGO_CUDA",
    "FiltroPyCUDA",
    "FiltroMediaPyCUDA",
    "FiltroNitidezPyCUDA",
    "FiltroLaplacianoPyCUDA",
    "crear_filtro",
    "crear_modulo",
]
