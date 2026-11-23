"""API Flask. El procesamiento y las reservas CUDA pertenecen al hilo GPUWorker."""
import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid
from flask import Flask, g, jsonify, render_template, request, send_file, url_for
from werkzeug.exceptions import HTTPException
from .core import GPUBackend, Limits, ProcessingError, parameters
from .worker import GPUWorker


def create_app(config=None, backend_factory=GPUBackend):
    app = Flask(__name__)
    app.config.update(RESULT_ROOT=os.getenv('RESULT_ROOT', '/tmp/practica3'),
        MAX_IMAGE_BYTES=int(os.getenv('MAX_IMAGE_MB', '40'))*1024*1024,
        MAX_PIXELS=int(os.getenv('MAX_PIXELS', '36000000')),
        MAX_SIDE=int(os.getenv('MAX_SIDE', '8000')),
        RESULT_TTL=int(os.getenv('RESULT_TTL', '3600')),
        MAX_STORED_RESULTS=int(os.getenv('MAX_STORED_RESULTS', '24')),
        GPU_CAPACITY=int(os.getenv('GPU_CAPACITY', '2')))
    if config:
        app.config.update(config)
    app.config['MAX_CONTENT_LENGTH'] = app.config['MAX_IMAGE_BYTES'] + 1024*1024
    root = Path(app.config['RESULT_ROOT']); root.mkdir(parents=True, exist_ok=True)
    limits = Limits(max_bytes=app.config['MAX_IMAGE_BYTES'],
                    max_pixels=app.config['MAX_PIXELS'], max_side=app.config['MAX_SIDE'])
    worker = GPUWorker(limits, app.config['GPU_CAPACITY'], backend_factory)
    app.extensions['gpu_worker'] = worker
    atexit.register(worker.close)
    storage_lock = threading.RLock()
    active = set()

    def cleanup():
        # Se llama bajo storage_lock. No elimina solicitudes en proceso.
        now = time.time()
        folders = sorted((p for p in root.iterdir() if p.is_dir() and p.name not in active),
                         key=lambda p: p.stat().st_mtime)
        for p in folders[:]:
            if now-p.stat().st_mtime > app.config['RESULT_TTL']:
                shutil.rmtree(p, ignore_errors=True); folders.remove(p)
        while len(folders) >= app.config['MAX_STORED_RESULTS']:
            shutil.rmtree(folders.pop(0), ignore_errors=True)

    def maintenance():
        while not stop.wait(60):
            with storage_lock:
                cleanup()
    stop = threading.Event()
    cleaner = threading.Thread(target=maintenance, daemon=True, name='result-cleaner'); cleaner.start()
    atexit.register(stop.set)
    app.extensions['cleaner_stop'] = stop

    def log(record):
        print(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(), **record),
                         ensure_ascii=False), flush=True)

    @app.before_request
    def identify():
        g.request_id = uuid.uuid4().hex
        g.started = time.perf_counter()

    @app.after_request
    def headers(response):
        response.headers['X-Request-ID'] = g.request_id
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(ProcessingError)
    def controlled(exc):
        log(dict(request_id=g.request_id, status='error', http_status=exc.status,
                 code=exc.code, detail=str(exc)))
        response = jsonify(request_id=g.request_id, status='error', code=exc.code, message=str(exc))
        if exc.status == 429:
            response.headers['Retry-After'] = '1'
        return response, exc.status

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return controlled(ProcessingError('Solicitud HTTP invalida: '+str(exc.description), exc.code, 'http_error'))

    @app.errorhandler(Exception)
    def unexpected(exc):
        app.logger.exception('Error solicitud %s', g.request_id)
        return controlled(ProcessingError('Fallo de procesamiento. Consulte el registro con el identificador.', 500, 'interno'))

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/health')
    def health():
        ready = worker.ready.is_set() and not worker.error
        return jsonify(status='ready' if ready else 'unavailable', gpu=worker.info,
                       diagnostic=worker.error, capacity=worker.capacity,
                       limits=dict(max_bytes=limits.max_bytes, max_pixels=limits.max_pixels,
                                   max_side=limits.max_side),
                       code_commit=os.getenv('CODE_COMMIT','no_registrado')), 200 if ready else 503

    @app.post('/api/process')
    def convolve():
        f, k, threads, raw = parameters(request.form)
        upload = request.files.get('image')
        if upload is None:
            raise ProcessingError('Falta el campo image.', 400)
        data = upload.stream.read(limits.max_bytes+1)
        if len(data) > limits.max_bytes:
            raise ProcessingError('Archivo supera el limite configurado.', 413)
        folder = root/g.request_id
        with storage_lock:
            cleanup(); folder.mkdir(); active.add(g.request_id)
        success = False
        try:
            metrics = worker.submit(data, f, k, threads, raw, folder).result()
            metrics.update(request_id=g.request_id, status='ok')
            (folder/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
            success = True
            log(metrics)
            return jsonify(metrics=metrics, request_id=g.request_id, status='ok',
                result_url=url_for('artifact', rid=g.request_id, kind='result'),
                original_url=url_for('artifact', rid=g.request_id, kind='original'),
                raw_url=url_for('artifact', rid=g.request_id, kind='raw') if raw else None)
        finally:
            with storage_lock:
                active.discard(g.request_id)
                if not success:
                    shutil.rmtree(folder, ignore_errors=True)

    @app.get('/api/results/<rid>/<kind>')
    def artifact(rid, kind):
        try:
            if uuid.UUID(hex=rid).hex != rid:
                raise ValueError()
        except ValueError:
            raise ProcessingError('Identificador invalido.', 404)
        mapping = dict(result=('result.png','image/png'), original=('original.png','image/png'),
                       raw=('raw.npy','application/octet-stream'))
        if kind not in mapping:
            raise ProcessingError('Recurso desconocido.', 404)
        name, mime = mapping[kind]
        with storage_lock:
            path = root/rid/name
            if not path.is_file() or time.time()-path.parent.stat().st_mtime > app.config['RESULT_TTL']:
                raise ProcessingError('Resultado inexistente o expirado; vuelva a procesar.', 404)
            # Leer bajo lock impide borrar durante una descarga.
            from io import BytesIO
            content = BytesIO(path.read_bytes())
        return send_file(content, mimetype=mime, download_name=rid+'_'+name,
                         as_attachment=request.args.get('download') == '1')
    return app
