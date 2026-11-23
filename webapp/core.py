"""Procesamiento compartido por HTTP y CLI; reutiliza kernels de la practica 02."""
from dataclasses import dataclass
from io import BytesIO
import hashlib
import time
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class ProcessingError(Exception):
    def __init__(self, message, status=422, code='entrada_invalida'):
        super().__init__(message)
        self.status, self.code = status, code


@dataclass(frozen=True)
class Limits:
    max_bytes: int = 40 * 1024 * 1024
    max_pixels: int = 36_000_000
    max_side: int = 8000
    raw_max_pixels: int = 1_000_000


def parameters(form):
    f = form.get('filter', 'media')
    try:
        k, threads = int(form.get('mask', 3)), int(form.get('threads', 256))
    except (ValueError, TypeError):
        raise ProcessingError('mask y threads deben ser enteros.', 400)
    if f not in ('media', 'nitidez', 'laplaciano'):
        raise ProcessingError('Filtro permitido: media, nitidez o laplaciano.', 400)
    if k not in ((3, 9, 21) if f == 'media' else (3,)):
        raise ProcessingError('Mascara incompatible con el filtro.', 400)
    if threads not in (128, 256, 512):
        raise ProcessingError('threads debe ser 128, 256 o 512.', 400)
    raw = form.get('raw', '0')
    if raw not in ('0', '1'):
        raise ProcessingError('raw debe ser 0 o 1.', 400)
    return f, k, threads, raw == '1'


def decode(data, limits=Limits()):
    if not data or len(data) > limits.max_bytes:
        raise ProcessingError('Imagen vacia o mayor que el limite de archivo.', 413, 'limite_archivo')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as im:
                if im.format not in ('JPEG', 'PNG', 'PPM', 'BMP', 'TIFF'):
                    raise ProcessingError('Formato real no permitido. Use JPG, PNG, PGM, BMP o TIFF de 8 bits.')
                if (im.width * im.height > limits.max_pixels or
                        max(im.size) > limits.max_side or min(im.size) < 1):
                    raise ProcessingError('Dimensiones exceden los limites configurados.', 413, 'limite_dimensiones')
                if getattr(im, 'n_frames', 1) != 1 or im.mode not in ('1','L','LA','P','RGB','RGBA','CMYK'):
                    raise ProcessingError('Use una imagen estatica de 8 bits por canal.')
                im.verify()
            with Image.open(BytesIO(data)) as im:
                im = ImageOps.exif_transpose(im)
                if im.mode in ('LA', 'P', 'RGBA'):
                    rgba = im.convert('RGBA')
                    im = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
                return np.array(im.convert('L'), dtype=np.float32, order='C')
    except ProcessingError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ProcessingError('El contenido no es una imagen valida y segura.') from exc


def png_bytes(array, filtro=None):
    # Misma regla que scripts/comun.py; validar float32 ANTES de visualizar.
    a = np.abs(array) if filtro == 'laplaciano' else array
    visual = np.floor(np.clip(a, 0, 255) + .5).astype(np.uint8)
    b = BytesIO()
    Image.fromarray(visual).save(b, format='PNG')
    return b.getvalue()


class GPUBackend:
    """Usar y cerrar exclusivamente en el hilo que crea el contexto."""
    def __init__(self):
        self.context = self.module = None
        try:
            import pycuda
            import pycuda.driver as cuda
            from src.pycuda.filtros_gpu import crear_modulo
            self.cuda = cuda
            t = time.perf_counter()
            cuda.init()
            if cuda.Device.count() == 0:
                raise RuntimeError('No hay dispositivos NVIDIA visibles.')
            dev = cuda.Device(0)
            self.context = dev.make_context()
            init_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            self.module = crear_modulo()
            compile_ms = (time.perf_counter() - t) * 1000
            self.info = dict(backend='pycuda_real', gpu=dev.name(),
                gpu_total_bytes=dev.total_memory(), compute_capability=dev.compute_capability(),
                driver_api=cuda.get_driver_version(), cuda_headers=cuda.get_version(),
                pycuda=pycuda.VERSION_TEXT, context_init_ms=init_ms,
                compile_load_ms=compile_ms, memory='NumPy paginable, copias sincronas',
                compiler_cache='SourceModule cache por defecto; no se fuerza compilacion fria')
        except Exception as exc:
            self.close()
            raise ProcessingError('No se pudo iniciar PyCUDA. Compruebe GPU NVIDIA, controlador, '
                'nvcc y --gpus all. Detalle: ' + str(exc), 503, 'gpu_no_disponible') from exc

    def run(self, entry, filtro, k, threads):
        from src.pycuda.filtros_gpu import crear_filtro
        free, _ = self.cuda.mem_get_info()
        if entry.nbytes * 2 + 256 * 1024 * 1024 > free:
            raise ProcessingError('Memoria GPU insuficiente para esta imagen.', 507, 'memoria_gpu')
        gpu = None
        try:
            t = time.perf_counter()
            gpu = crear_filtro(self.module, entry, filtro, k, threads)
            allocation_ms = (time.perf_counter() - t) * 1000
            times = gpu.ejecutar()
            # Mantener vista de salida: cerrar libera buffers de dispositivo, no el NumPy.
            result = gpu.salida
            if not np.isfinite(result).all():
                raise ProcessingError('GPU produjo NaN o infinito.', 500, 'resultado_gpu')
            return result, dict(times, allocation_ms=allocation_ms, blocks=gpu.bloques)
        except self.cuda.MemoryError as exc:
            raise ProcessingError('Fallo de reserva en GPU.', 507, 'memoria_gpu') from exc
        except self.cuda.Error as exc:
            raise ProcessingError('Error CUDA: ' + str(exc), 500, 'error_cuda') from exc
        finally:
            if gpu is not None:
                gpu.cerrar()

    def close(self):
        self.module = None
        if self.context is not None:
            self.context.pop()
            self.context.detach()
            self.context = None


def process(data, filtro, k, threads, backend, limits=Limits(), raw=False):
    start = time.perf_counter()
    entry = decode(data, limits)
    if raw and entry.size > limits.raw_max_pixels:
        raise ProcessingError('La salida float32 de validacion se limita a un millon de pixeles.', 413)
    decode_ms = (time.perf_counter() - start) * 1000
    raw_result, times = backend.run(entry, filtro, k, threads)
    t = time.perf_counter()
    result_png, original_png = png_bytes(raw_result, filtro), png_bytes(entry)
    raw_bytes = None
    if raw:
        b = BytesIO(); np.save(b, raw_result, allow_pickle=False); raw_bytes = b.getvalue()
    encode_ms = (time.perf_counter() - t) * 1000
    metrics = dict(times, decode_prepare_ms=decode_ms, encode_ms=encode_ms,
        processing_ms=(time.perf_counter()-start)*1000,
        width=entry.shape[1], height=entry.shape[0], filter=filtro, mask=k, threads=threads,
        sha256_input=hashlib.sha256(data).hexdigest(),
        sha256_float32=hashlib.sha256(raw_result.tobytes()).hexdigest(),
        unit='ms', backend=backend.info['backend'])
    return dict(png=result_png, original=original_png, raw=raw_bytes, metrics=metrics)
