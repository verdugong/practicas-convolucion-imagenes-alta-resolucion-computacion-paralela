# API HTTP

Base de ejemplo: `http://localhost:8000`. No usar 0.0.0.0 como URL del navegador.

## GET /api/health

200 cuando el hilo GPU creó contexto y cargó/compiló los kernels; 503 durante
inicialización o si no existe GPU. Devuelve diagnóstico, versión PyCUDA/CUDA,
nombre y memoria GPU, tiempos de inicialización/compilación y límites. El estado
no sustituye una ejecución numérica: usar `python -m tools.diagnose`.

## POST /api/process

Cuerpo `multipart/form-data`:

| Campo | Valores | Predeterminado |
|---|---|---|
| image | Archivo real JPG, PNG, PGM, BMP o TIFF, estático, 8 bits | Obligatorio |
| filter | media, nitidez, laplaciano | media |
| mask | media: 3/9/21; otros: 3 | 3 |
| threads | 128/256/512 | 256 |
| raw | 0/1; float32 NPY solo para validación hasta 1 MP | 0 |

No se redimensiona la imagen ni se cambia silenciosamente la máscara. EXIF se
aplica y transparencia se compone sobre blanco antes de convertir a gris.
Límites predeterminados: 40 MiB de archivo, 36 MP, lado máximo 8000 px. Se verifica
el contenido real con Pillow, no se confía en la extensión ni en MIME del cliente.
El límite HTTP total permite 1 MiB adicional para el formulario; el archivo tiene
su límite separado de 40 MiB. Límites se aplican antes de reservar en GPU.
También se comprueba memoria GPU libre: dos buffers float32 y margen de 256 MiB.
Para 6000² los dos buffers ocupan unos 288 MB decimales; el host usa más memoria
por conversión, PNG y verificaciones. Se recomienda al menos 8 GB de RAM y GPU de
4 GB o mayor para las pruebas, ajustando límites tras diagnosticar el equipo.

Ejemplo CMD de Windows (usar curl.exe si se ejecuta en PowerShell):

```bat
curl.exe -F "image=@data/preparadas/imagen_4000.pgm" -F "filter=media" -F "mask=9" -F "threads=256" http://localhost:8000/api/process
```

Respuesta 200 (valores de tiempos dependen de la ejecución):

```json
{"status":"ok","request_id":"UUID_HEX","result_url":"/api/results/UUID_HEX/result",
 "original_url":"/api/results/UUID_HEX/original","raw_url":null,
 "metrics":{"filter":"media","mask":9,"threads":256,"unit":"ms"}}
```

`metrics` incluye width, height, blocks, hash de entrada/float32, H2D, kernel,
D2H, total GPU, reservas, preparación, codificación, escritura, procesamiento del
servidor y espera. El CSV cliente añade latencia hasta descargar completamente el PNG.
La respuesta JSON por sí sola NO completa la medición del cliente.

## GET /api/results/{id}/result

Devuelve PNG. Agregar `?download=1` para descarga como adjunto. `original` devuelve
la entrada gris efectivamente calculada y `raw` devuelve NPY float32 cuando fue
solicitado. El NPY conserva signo y valores fuera de [0,255]. PNG aplica abs al
laplaciano, clip a [0,255] y redondeo `floor(x+0.5)`, igual que práctica 02.

Un UUID único y una carpeta independiente por solicitud evitan sobreescritura y
mezcla. No se utiliza el nombre del archivo subido para escribir en disco.
Los identificadores no son autenticación: esta implementación es para laboratorio
con imágenes no confidenciales. Para acceso público permanente se requieren
HTTPS/autenticación y control de acceso, fuera del alcance de esta demostración.

## Errores y política de atención

| HTTP | Caso |
|---|---|
| 400 | Falta image, filtro/máscara/hilos/raw inválidos |
| 413 | Archivo, píxeles o dimensiones superan el límite |
| 422 | Contenido inválido, formato o profundidad no admitidos |
| 429 | Cola llena; Retry-After: 1 |
| 503 | GPU ausente, iniciando o no accesible |
| 507 | Memoria GPU insuficiente |
| 500 | Error CUDA o procesamiento inesperado |
| 404 | Recurso desconocido, identificador inválido o resultado vencido |

Todos devuelven JSON con request_id, status, code y message; el identificador
también está en `X-Request-ID`. Cada fallo queda en logs JSON de Gunicorn.
Un único hilo/contexto GPU atiende FIFO; capacidad total 2: una activa y una en
espera. La tercera simultánea se rechaza si ambas plazas están ocupadas. El
tiempo de espera se mide con `perf_counter` en ese mismo servidor. Hay 8 hilos
HTTP y UN proceso Gunicorn; no aumentar workers ni usar preload/reload.

Resultados temporales: TTL 3600 s, máximo 24 resultados. Un limpiador cada 60 s y
la admisión de nuevas solicitudes eliminan los más antiguos; no se eliminan
trabajos activos. Tras TTL o por límite de capacidad se devuelve 404. Descargar
lo importante en seguida; tools.client guarda evidencia fuera de esos temporales.
Variables: MAX_IMAGE_MB, MAX_PIXELS, MAX_SIDE, GPU_CAPACITY, RESULT_TTL,
MAX_STORED_RESULTS y RESULT_ROOT. Si se cambian, registrar valores en evidencia.
