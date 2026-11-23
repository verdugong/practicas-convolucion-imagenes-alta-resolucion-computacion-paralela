"""Python secuencial: piloto, calentamiento, CSV incremental y reanudacion."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.python.filtros_python import crear_filtro_cpu
from comun import (COLUMNAS, TOLERANCIA, ahora, cargar_imagen, cargar_referencia,
                   errores, guardar_json, guardar_visual, interrupciones, leer_filas)


def medir(entrada, referencia, filtro, k, reps, prefijo, reanudar=False):
    if not 5 <= reps <= 1000:
        raise ValueError("Se requieren entre 5 y 1000 repeticiones.")
    prefijo = Path(prefijo)
    archivo_csv = Path(str(prefijo) + "_python_raw.csv")
    estado_path = Path(str(prefijo) + "_python_estado.json")
    if archivo_csv.exists() and not reanudar:
        raise FileExistsError("Prefijo existente: use --resume o un prefijo nuevo.")
    prefijo.parent.mkdir(parents=True, exist_ok=True)
    anteriores = leer_filas(archivo_csv)
    for i, r in enumerate(anteriores, 1):
        if (r["implementacion"] != "PYTHON" or r["filtro"] != filtro
            or int(r["mascara"]) != k or int(r["ancho"]) != entrada.shape[1]
            or int(r["alto"]) != entrada.shape[0] or int(r["repeticion"]) != i
            or r.get("estado") != "completada" or r["validado"] != "true"
            or not np.isfinite(float(r["total_ms"])) or float(r["total_ms"]) <= 0
            or not np.isfinite(float(r["error_max"]))
            or not 0 <= float(r["error_max"]) <= TOLERANCIA):
            raise ValueError("CSV previo inconsistente; no se mezcla con esta ejecucion.")
    if len(anteriores) > reps:
        raise ValueError("CSV con mas repeticiones que las solicitadas.")
    f_cpu = crear_filtro_cpu(filtro, k)
    salida = f_cpu.preparar(entrada)
    estado = {"estado": "en_curso", "inicio_utc": ahora(), "filtro": filtro,
              "mascara": k, "forma": list(entrada.shape), "repeticiones_objetivo": reps,
              "repeticiones_completas": len(anteriores), "fase": "preparacion",
              "alcance_cpu": "solo aplicar: recorridos y acumulacion float32; salida y mascara preasignadas"}
    if estado_path.exists():
        anterior = json.loads(estado_path.read_text(encoding="utf-8"))
        estado["sesiones_previas"] = anterior.get("sesiones_previas", []) + [
            {k: v for k, v in anterior.items() if k != "sesiones_previas"}]
    inicio_sesion = perf_counter()
    inicio_fase = inicio_sesion
    def fase(nombre):
        nonlocal inicio_fase
        estado["fase"] = nombre
        estado["fase_inicio_utc"] = ahora()
        inicio_fase = perf_counter()
        guardar_json(estado_path, estado)
    try:
        fase("piloto_64")
        piloto = np.ascontiguousarray(entrada[:64, :64])
        prueba = f_cpu.preparar(piloto)
        t0 = perf_counter()
        f_cpu.aplicar(piloto, prueba)
        piloto_s = perf_counter() - t0
        estimacion = piloto_s * entrada.size / piloto.size
        estado["piloto"] = {"pixeles": int(piloto.size), "tiempo_s": piloto_s,
                             "estimacion_por_repeticion_s": estimacion,
                             "estimacion_sesion_s": estimacion * (1 + reps - len(anteriores)),
                             "advertencia": "ESTIMACION, no medicion de imagen completa; bordes/caches afectan"}
        print(f"Piloto: {piloto_s:.3f} s. ESTIMACION por imagen: {estimacion:.1f} s; "
              f"sesion restante: {estado['piloto']['estimacion_sesion_s']:.1f} s.", flush=True)
        fase("calentamiento")
        print("Python: calentamiento completo en curso (no cuenta como repeticion).", flush=True)
        f_cpu.aplicar(entrada, salida)
        if errores(referencia, salida)[0] > TOLERANCIA:
            raise ValueError("El calentamiento no coincide con C++.")
        estado["calentamiento_s"] = perf_counter() - inicio_fase
        with archivo_csv.open("a" if anteriores else "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNAS)
            if not anteriores:
                w.writeheader()
                f.flush()
            for r in range(len(anteriores) + 1, reps + 1):
                estado["repeticion_en_curso"] = r
                fase("convolucion")
                t0 = perf_counter()
                f_cpu.aplicar(entrada, salida)
                ms = (perf_counter() - t0) * 1000
                fase("validacion")
                emax, emedio = errores(referencia, salida)
                valido = emax <= TOLERANCIA
                w.writerow(dict(implementacion="PYTHON", filtro=filtro, mascara=k,
                    ancho=entrada.shape[1], alto=entrada.shape[0], hilos=0, bloques=0,
                    repeticion=r, cpu_ms=ms, total_ms=ms, error_max=emax,
                    error_medio=emedio, tolerancia=TOLERANCIA, validado=str(valido).lower(),
                    unidad="ms", estado="completada" if valido else "fallida"))
                f.flush()
                if not valido:
                    raise ValueError("Python difiere de C++; CSV del fallo conservado.")
                estado["repeticiones_completas"] = r
                guardar_json(estado_path, estado)
                print(f"Python {r}/{reps}: {ms:.6f} ms, error CPU={emax}", flush=True)
        guardar_visual(str(prefijo) + "_python.pgm", salida, filtro)
        estado["estado"] = "completada"
        estado["fase"] = "finalizada"
    except KeyboardInterrupt:
        estado["estado"] = "no_completada"
        estado["motivo"] = "interrupcion recibida; consultar motivo del coordinador"
        estado["fase_incompleta_transcurrido_s"] = perf_counter() - inicio_fase
        raise
    except Exception as exc:
        estado["estado"] = "fallida"
        estado["motivo"] = str(exc)
        raise
    finally:
        estado["sesion_transcurrido_s"] = perf_counter() - inicio_sesion
        estado["fin_utc"] = ahora()
        guardar_json(estado_path, estado)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--filter", choices=["media", "nitidez", "laplaciano"], default="media")
    p.add_argument("--mask", type=int, default=3)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--reference-prefix", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    interrupciones()
    entrada = cargar_imagen(a.input)
    ref = cargar_referencia(str(a.reference_prefix) + "_cpu.f32", entrada.shape)
    medir(entrada, ref, a.filter, a.mask, a.reps, a.output, a.resume)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Python interrumpido: se conservaron repeticiones completas y estado.", flush=True)
        sys.exit(130)
