"""Una cola acotada y un hilo CUDA; FIFO, sin compartir el contexto entre hilos."""
from concurrent.futures import Future
from pathlib import Path
import queue
import threading
import time
from .core import GPUBackend, ProcessingError, process


class GPUWorker:
    def __init__(self, limits, capacity=2, factory=GPUBackend):
        self.limits = limits
        self.jobs = queue.Queue()
        self.slots = threading.BoundedSemaphore(capacity)
        self.capacity, self.factory = capacity, factory
        self.ready = threading.Event()
        self.error, self.info = None, {}
        self._closed = False
        self.thread = threading.Thread(target=self._loop, name='gpu-owner', daemon=True)
        self.thread.start()

    def _loop(self):
        backend = None
        try:
            backend = self.factory()
            self.info = backend.info
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.ready.set()
        if backend is None:
            return
        try:
            while True:
                job = self.jobs.get()
                if job is None:
                    break
                future, submitted, args, folder = job
                try:
                    started = time.perf_counter()
                    wait_ms = (started - submitted) * 1000
                    data, filtro, k, threads, raw = args
                    result = process(data, filtro, k, threads, backend, self.limits, raw=raw)
                    write = time.perf_counter()
                    for key, name in [('png','result.png'),('original','original.png'),('raw','raw.npy')]:
                        if result[key] is not None:
                            (folder/name).write_bytes(result[key])
                    m = result['metrics']
                    m.update(write_ms=(time.perf_counter()-write)*1000, queue_ms=wait_ms,
                        server_processing_ms=(time.perf_counter()-started)*1000)
                    if 'cold_first_job' not in self.info:
                        self.info['cold_first_job'] = dict(m,
                            context_init_ms=self.info.get('context_init_ms'),
                            compile_load_ms=self.info.get('compile_load_ms'),
                            scope='Primera solicitud del proceso. Inicializacion separada; no incluye arranque Docker ni subida HTTP.')
                    future.set_result(m)
                except Exception as exc:
                    future.set_exception(exc)
                finally:
                    self.slots.release()
        finally:
            backend.close()

    def submit(self, data, filtro, k, threads, raw, folder):
        if not self.ready.wait(120):
            raise ProcessingError('GPU inicializando. Reintente despues.', 503, 'gpu_iniciando')
        if self.error:
            raise ProcessingError(self.error, 503, 'gpu_no_disponible')
        if not self.slots.acquire(blocking=False):
            raise ProcessingError('Capacidad alcanzada: una solicitud activa y una en espera.', 429, 'cola_llena')
        future = Future()
        self.jobs.put((future, time.perf_counter(), (data, filtro, k, threads, raw), Path(folder)))
        return future

    def close(self):
        if not self._closed:
            self._closed = True
            if self.thread.is_alive():
                self.jobs.put(None)
                self.thread.join(timeout=10)
