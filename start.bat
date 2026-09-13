@echo off
cd /d "%~dp0"

echo Clearing cache...
REM Delete __pycache__ directories in program folder only (skip .venv)
for /d /r "program" %%d in (__pycache__) do @if exist "%%d" (
    rmdir /s /q "%%d"
)

REM Delete program/cache directory
if exist "program\cache" (
    echo Deleting: program\cache
    rmdir /s /q "program\cache"
)

echo Cache cleared. Starting application...
echo.

call .venv\Scripts\activate
python program\app.py
echo.
echo Server has exited. Press any key to close this window.
pause >nul
