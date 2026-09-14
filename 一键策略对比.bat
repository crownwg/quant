@echo off
chcp 936 >nul
cd /d "%~dp0"
title 一键策略对比

echo.
echo   ============================================================
echo      一键策略对比
echo      给它一只或几只股票，自动跑完全样本 + 样本外，出一份报告
echo   ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo   [错误] 没找到 .venv\Scripts\python.exe
    echo   请确认这个 bat 文件和 .venv 在同一个目录下
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" 一键策略对比.py

echo.
pause
