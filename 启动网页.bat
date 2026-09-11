@echo off
chcp 936 >nul
cd /d "%~dp0"
title Quant Web Console

curl -s -m 2 http://127.0.0.1:8765/api/health >nul 2>&1
if not errorlevel 1 (
    echo Server is already running. Opening browser...
    start "" http://127.0.0.1:8765/
    timeout /t 2 /nobreak >nul
    exit
)

echo ==========================================================
echo    Quant Web Console   (local, no internet needed)
echo ==========================================================
echo.
echo    URL  : http://127.0.0.1:8765/
echo    Stop : close the "quant-web" window in your taskbar
echo.
echo    Starting server... browser opens in 4 seconds
echo ==========================================================
echo.

start "quant-web" /min "%~dp0.venv\Scripts\python.exe" -m quant.web
timeout /t 4 /nobreak >nul
start "" http://127.0.0.1:8765/
echo.
echo Done. This window closes by itself.
timeout /t 3 /nobreak >nul
exit
