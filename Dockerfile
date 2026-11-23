# devel es intencional: SourceModule necesita nvcc en TIEMPO DE EJECUCION.
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04
ARG CODE_COMMIT=no_registrado
LABEL org.opencontainers.image.title="Practica 03 - PyCUDA Web" \
      org.opencontainers.image.revision=$CODE_COMMIT
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    CUDA_ROOT=/usr/local/cuda CODE_COMMIT=$CODE_COMMIT \
    RESULT_ROOT=/app/runtime GPU_CAPACITY=2 HOME=/home/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv build-essential ca-certificates git fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir pip==24.3.1 setuptools==75.6.0 wheel==0.45.1 \
    && LIBRARY_PATH=/usr/local/cuda/lib64/stubs pip install --no-cache-dir -r requirements-web.txt \
    && pip check && pip freeze > /opt/environment-freeze.txt
COPY . .
RUN python -m tools.prepare_data && python -m tools.verify_sources \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/runtime /app/evidence \
    && chown -R app:app /app/runtime /app/evidence /home/app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4)" || exit 1
# No preload: el contexto CUDA nace DESPUES del fork de Gunicorn.
CMD ["gunicorn","--config","gunicorn.conf.py","webapp.app:create_app()"]
