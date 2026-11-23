"""Coordinacion comun de memoria y ejecucion para los filtros PyCUDA."""

from time import perf_counter
import numpy as np


class FiltroPyCUDA:
    """Clase base: reserva recursos una vez y mide cada ejecucion."""

    nombre_kernel = None
    k_permitidas = (3,)

    def __new__(cls, modulo, entrada, filtro, k, hilos):
        if cls is FiltroPyCUDA:
            from .laplaciano_gpu import FiltroLaplacianoPyCUDA
            from .media_gpu import FiltroMediaPyCUDA
            from .nitidez_gpu import FiltroNitidezPyCUDA

            clases = {
                "media": FiltroMediaPyCUDA,
                "nitidez": FiltroNitidezPyCUDA,
                "laplaciano": FiltroLaplacianoPyCUDA,
            }
            try:
                cls = clases[filtro]
            except KeyError as exc:
                raise ValueError("Filtro invalido.") from exc
        return super().__new__(cls)

    def __init__(self, modulo, entrada, filtro, k, hilos):
        import pycuda.driver as cuda

        if (
            entrada.ndim != 2
            or entrada.dtype != np.float32
            or not entrada.flags.c_contiguous
        ):
            raise ValueError("Entrada requerida: matriz float32 contigua.")
        if k not in self.k_permitidas:
            raise ValueError("Mascara invalida para el filtro seleccionado.")
        if hilos not in (128, 256, 512):
            raise ValueError("Hilos permitidos: 128, 256 o 512.")
        alto, ancho = entrada.shape
        if min(alto, ancho) <= 0:
            raise ValueError("La imagen esta vacia.")
        self.bloques = (entrada.size + hilos - 1) // hilos
        device = cuda.Context.get_device()
        if (
            hilos > device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
            or hilos > device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_X)
            or self.bloques > device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)
        ):
            raise ValueError("Configuracion excede los limites de la GPU.")
        self.cuda = cuda
        self.entrada = entrada
        self.salida = np.empty_like(entrada)
        self.d_entrada = cuda.mem_alloc(entrada.nbytes)
        try:
            self.d_salida = cuda.mem_alloc(entrada.nbytes)
        except Exception:
            self.d_entrada.free()
            raise
        self.inicio, self.fin = cuda.Event(), cuda.Event()
        self.filtro = filtro
        self.kernel = modulo.get_function(self.nombre_kernel)
        self.args = (self.d_entrada, self.d_salida, np.int32(ancho), np.int32(alto))
        if filtro == "media":
            self.args += (np.int32(k), np.float32(1) / np.float32(k * k))
        self.block, self.grid = (hilos, 1, 1), (self.bloques, 1)

    def ejecutar(self):
        """Ejecutar H2D, kernel y D2H, devolviendo sus tiempos en milisegundos."""
        cuda = self.cuda
        cuda.Context.synchronize()
        t0 = perf_counter()
        cuda.memcpy_htod(self.d_entrada, self.entrada)
        cuda.Context.synchronize()
        t1 = perf_counter()
        self.inicio.record()
        self.kernel(*self.args, block=self.block, grid=self.grid)
        self.fin.record()
        self.fin.synchronize()
        t2 = perf_counter()
        cuda.memcpy_dtoh(self.salida, self.d_salida)
        t3 = perf_counter()
        return {
            "h2d_ms": (t1 - t0) * 1000,
            "kernel_ms": self.inicio.time_till(self.fin),
            "d2h_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        }

    def cerrar(self):
        self.d_entrada.free()
        self.d_salida.free()
        self.inicio = self.fin = self.kernel = None
        self.args = ()
