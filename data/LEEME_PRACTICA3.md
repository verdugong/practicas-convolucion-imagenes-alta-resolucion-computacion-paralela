# Datos de práctica 03

Los .pgm.gz contienen los PGM recibidos sin cambios. Cada archivo comprimido pesa
menos de 25 MiB y puede subirse por arrastre a GitHub. No es necesario subir el PGM
6000 sin comprimir (supera el límite web de 25 MiB).

Desde la raíz del proyecto ejecutar:

```bat
python -m tools.prepare_data
python -m tools.verify_sources
```

El Dockerfile ya ejecuta esos pasos al construir. No usa JPEG para reconstruir
los PGM ni depende de la versión de Pillow para restaurar datos: gzip recupera
los mismos bytes y comprueba SHA-256 de docs/data_practica3.json.
La fuente/autoría de los JPEG y permisos de uso están pendientes en el manifiesto
original: el grupo debe completarlos; los hashes no demuestran permiso de uso.
