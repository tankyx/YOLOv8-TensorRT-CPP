@echo off
setlocal enabledelayedexpansion
set PRECISION=%1
if "%PRECISION%"=="" set PRECISION=half
if not "%PRECISION%"=="half" if not "%PRECISION%"=="float" (
    echo Usage: run_valo.bat [half^|float]
    echo   half  — FP16 (default, recommended)
    echo   float — FP32
    exit /b 1
)

cd /d "C:\Users\tanguy\Documents\GitHub\YOLOv8-TensorRT-CPP\build2\bin\Release"

echo Starting YOLOv8-TensorRT for Valorant...
echo Config: config_valo.ini
echo Precision: %PRECISION%
echo Supports: YOLOv8, YOLOv11, YOLOv26 (auto-detect)
echo Press Ctrl+C in this window to stop.
echo.

REM Create a temp config with the requested precision, leaving the original untouched.
copy /y config_valo.ini config_valo_temp.ini > nul
powershell -Command "(gc config_valo_temp.ini) -replace '^Precision = .*', 'Precision = %PRECISION%' | Out-File config_valo_temp.ini -Encoding ASCII"

detect_object_image.exe config_valo_temp.ini
del config_valo_temp.ini
