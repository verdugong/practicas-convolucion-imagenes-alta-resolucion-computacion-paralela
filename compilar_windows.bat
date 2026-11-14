@echo off
setlocal
cd /d "%~dp0"
where cl >nul 2>nul
if errorlevel 1 (
 echo Abrir x64 Native Tools Command Prompt for VS 2022 y ejecutar este archivo.
 exit /b 1
)
where nvcc >nul 2>nul
if errorlevel 1 (
 echo No se encontro nvcc. Revisar CUDA Toolkit y PATH.
 exit /b 1
)
python -c "import numpy, PIL, matplotlib, psutil, pycuda.driver"
if errorlevel 1 (
 echo Instalar dependencias en este Python: python -m pip install -r requirements.txt
 exit /b 1
)
python scripts\compilar.py
if errorlevel 1 exit /b 1
ctest --test-dir build -C Release -V > verificacion_gpu_laboratorio.txt 2>&1
if errorlevel 1 (
 type verificacion_gpu_laboratorio.txt
 exit /b 1
)
type verificacion_gpu_laboratorio.txt
findstr /C:"100%% tests passed" verificacion_gpu_laboratorio.txt >nul
if errorlevel 1 (
 echo Revisar el registro de pruebas antes de ejecutar experimentos.
 exit /b 1
)
echo Compilacion y pruebas terminadas. Ejecutar ejecutar_windows.bat
