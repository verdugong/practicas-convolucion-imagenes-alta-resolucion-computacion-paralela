"""Registro atomico, validacion numerica y esquema CSV de la practica 02."""
import csv
import hashlib
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

TOLERANCIA = 0.001
COLUMNAS = ["implementacion", "filtro", "ancho", "alto", "mascara", "hilos",
            "bloques", "repeticion", "cpu_ms", "h2d_ms", "kernel_ms", "d2h_ms",
            "total_ms", "error_max", "error_medio", "tolerancia", "validado",
            "unidad", "error_cuda_max", "error_cuda_medio", "validado_cuda", "estado"]


def ahora():
    return datetime.now(timezone.utc).isoformat()


def guardar_json(ruta, contenido):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temp = ruta.with_name(ruta.name + ".tmp")
    temp.write_text(json.dumps(contenido, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")
    temp.replace(ruta)


def sha256(ruta):
    h = hashlib.sha256()
    with Path(ruta).open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_imagen(ruta):
    with Path(ruta).open("rb") as f:
        if f.read(2) != b"P5":
            raise ValueError("Se requiere PGM binario P5.")
    with Image.open(ruta) as im:
        if im.mode != "L":
            raise ValueError("Se requiere gris de 8 bits.")
        return np.array(im, dtype=np.float32, order="C")


def cargar_referencia(ruta, forma):
    if Path(ruta).stat().st_size != int(np.prod(forma)) * 4:
        raise ValueError(f"Referencia incompleta: {ruta}")
    return np.fromfile(ruta, dtype="<f4").reshape(forma)


def errores(referencia, resultado):
    if referencia.shape != resultado.shape or not referencia.size:
        raise ValueError("Dimensiones incompatibles.")
    if not np.isfinite(referencia).all() or not np.isfinite(resultado).all():
        return float("inf"), float("inf")
    a, b = referencia.ravel(), resultado.ravel()
    maximo, suma = 0.0, 0.0
    for i in range(0, a.size, 1_000_000):
        dif = np.abs(a[i:i+1_000_000].astype(np.float64)
                     - b[i:i+1_000_000].astype(np.float64))
        maximo = max(maximo, float(dif.max()))
        suma += float(dif.sum())
    return maximo, suma / a.size


def guardar_visual(ruta, resultado, filtro):
    valores = np.abs(resultado) if filtro == "laplaciano" else resultado
    Image.fromarray(np.floor(np.clip(valores, 0, 255) + 0.5).astype(np.uint8)).save(ruta)


def interrupciones():
    """CTRL_BREAK_EVENT permite cerrar y guardar CSV en Windows."""
    def interrumpir(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrumpir)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, interrumpir)


def leer_filas(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        return []
    with ruta.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
