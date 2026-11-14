"""Compatibilidad: la implementacion vive en src/python/filtros_python.py."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.python.filtros_python import (  # noqa: E402,F401
    FiltroCPU,
    FiltroMediaCPU,
    FiltroNitidezCPU,
    FiltroLaplacianoCPU,
    crear_filtro_cpu,
)
