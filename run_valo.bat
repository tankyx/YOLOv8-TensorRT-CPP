@echo off
setlocal enabledelayedexpansion

:: Parse model variant from first argument
set VARIANT=%1
if "%VARIANT%"=="" set VARIANT=v8

:: Map variant to config file
set CONFIG=
if /i "%VARIANT%"=="v8"   set CONFIG=config_valo_v8.ini
if /i "%VARIANT%"=="v11s" set CONFIG=config_valo_v11s.ini
if /i "%VARIANT%"=="v11m" set CONFIG=config_valo_v11m.ini
if /i "%VARIANT%"=="v26m" set CONFIG=config_valo_v26m.ini
if /i "%VARIANT%"=="v26l" set CONFIG=config_valo_v26l.ini

if "%CONFIG%"=="" (
    echo Usage: run_valo.bat [v8^|v11s^|v11m^|v26m^|v26l]
    echo.
    echo   v8    YOLOv8 nano  ^(yolov8n_val_fp16.onnx^)
    echo   v11s  YOLOv11 small ^(model not trained yet^)
    echo   v11m  YOLOv11 medium ^(model not trained yet^)
    echo   v26m  YOLOv26 medium ^(model not trained yet^)
    echo   v26l  YOLOv26 large  ^(model not trained yet^)
    exit /b 1
)

cd /d "C:\Users\tanguy\Documents\GitHub\YOLOv8-TensorRT-CPP\build2\bin\Release"

echo ========================================
echo   YOLO-TensorRT — Valorant
echo   Variant: %VARIANT%
echo   Config:  %CONFIG%
echo ========================================
echo Press Ctrl+C in this window to stop.
echo.

detect_object_image.exe %CONFIG%
