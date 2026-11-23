# Matriz de cumplimiento del enunciado

Esta tabla distingue implementación preparada de evidencia que exige GPU/Docker.
No marca una ejecución pendiente como si ya estuviera aprobada.

| Requisito | Implementación / ubicación | Evidencia a obtener |
|---|---|---|
| Origen práctica 02 y referencia ejecutable | docs/origin_practica2.json, bundle, código previo conservado | URL/SHA publicados, verify_sources |
| Kernels propios, 5 combinaciones, float32, ceros, un píxel por hilo X | src/pycuda sin cambios; webapp/core.py | Hashes y validación real |
| 2000/4000/6000 e identificación | data/preparadas, hashes en origin_practica2.json | verify_sources en host/contenedor |
| Carga, filtro/máscara, original/resultado y descarga | webapp/templates, static | Captura y solicitud real |
| Servicio HTTP documentado y consumido por UI | app.py, docs/API.md, app.js | POST + GET resultado completo |
| Tiempos y mensajes; prevenir doble envío | UI deshabilita botón; métricas y errores | Capturas/interacción |
| Validar formato real, tamaño, píxeles y parámetros | core.decode/parameters, Limits y app MAX_CONTENT_LENGTH | validate_gpu invalid_inputs.json |
| UUID, no mezclar resultados, limpieza | app.py; carpeta por UUID, TTL y máximo 24 | isolation.json, tests TTL |
| Procesamiento separado de HTTP y ejecutable directo | core.process, tools.direct | direct.csv + service.csv |
| Contexto, compilación, memoria/copias y sincronización | GPUBackend + clases originales; arquitectura documentada | gpu_init.json, health/logs |
| Sin GPU, OOM, fallo de carga/proceso | errores 503/507/422/500 sin fallback | diagnose y errores registrados |
| Dockerfile/.dockerignore/versiones | raíz y requirements-web.txt | build.log e ID/digest real |
| Driver host frente CUDA del contenedor/runtime | README, GUIA, base devel | nvidia-smi, docker info, diagnose |
| Servidor sin depuración/recarga | Gunicorn 1 proceso/8 hilos, sin preload | logs y docker inspect |
| Construcción desde copia limpia, commit exacto | tools.build verifica Git limpio | evidence/build/image.json |
| Nube NVIDIA o alternativa local documentada | instrucciones ambas rutas | Ubicación/motivo, acceso desde otro cliente |
| Entorno CPU/RAM/GPU/VRAM/OS/driver/Docker/runtime/Python/PyCUDA/CUDA | tools.evidence + common.environment + health | deployment.json; completar datos no detectados |
| Matriz manual, bordes y píxeles no múltiplo | validate_gpu + tests previos | 45 casos + impulso |
| Equivalencia tres filtros directo-servicio antes de visualizar | raw NPY, error max/medio float64, tol absoluta 0.001 | numerical_validation.json |
| Evidencia visual de original y tres filtros | validate_gpu guarda mismas regiones 17x19, web/cliente PNG grande | PNG y capturas |
| Entradas inválidas y recuperación | validate_gpu + tests_web | invalid_inputs.json y petición posterior correcta |
| Aislamiento dos solicitudes identificables y límite de cola | validate_gpu concurrente + prueba determinista de semáforo | isolation.json/queue_burst.json/pytest |
| Integración host/contenedor/servicio media9 4000 | tools.direct y client (comparten configuración) | Los tres CSV o limitación host explícita |
| Resolución media9 2000/4000/6000 | plan en tools.direct | Kernel, servidor y latencia cliente |
| Filtros/máscaras 4000 | cinco combinaciones del plan, media9 compartida | service.csv |
| Concurrencia 1/2, cinco rondas y una petición/cliente | tools.client, barrera y ThreadPoolExecutor | Cada petición, round_ms, éxitos/s, cola, errores |
| Calentamiento y al menos 5 repeticiones | direct/client; fases separadas | CSV stable frente warmup/primera |
| H2D/kernel/D2H separados y memoria declarada | clase original: eventos + copias síncronas paginables | Campos y arquitectura |
| Servidor monotónico/cliente incluye descarga | core, worker, client.one | server_processing_ms / client_total_ms |
| Contexto/compilación fuera de estables y primera ejecución | GPUBackend.info + fase cold_first del caso | Primer arranque health/log/primera solicitud |
| CSV con factores, identificadores, estados, entorno/unidad | tools.common.FIELDS, direct/client | CSV llenados por ejecución real |
| Mediana, variabilidad y fallos | summarize: mediana/IQR, resumen y figuras | summary.csv/json, fallos sin ceros ficticios |
| Limitaciones documentadas y alternativa menor | --small + imagen_512 incluida; guía | Motivo/log/tiempo y series comunes |
| Informe, aportes y firmas, repo/datos/evidencia | tools.report borrador PDF + metadatos | Interpretación final y firmas del grupo |
| AVAC y cierre de recursos | sección de entrega de guía | PDF final; nube detenida si se usa |

No se incluyen capturas ni CSV inventados para llenar las celdas. El directorio
results conserva únicamente evidencia recibida de práctica 02. Los nuevos CSV se
generan en evidence y deben corresponder a la imagen y commit medidos.
