@echo off
setlocal
cd /d "%~dp0"
python scripts\experimentos.py --exe build\practica1.exe --data data\preparadas --reps 5 %*
exit /b %errorlevel%
