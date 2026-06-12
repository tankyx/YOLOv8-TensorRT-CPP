@echo off
setlocal enabledelayedexpansion

:: Parse model variant from first argument
set VARIANT=%1
if "%VARIANT%"=="" set VARIANT=v8

:: Map variant to config file
set CONFIG=
if /i "%VARIANT%"=="v8"   set CONFIG=config_cs2_v8.ini
if /i "%VARIANT%"=="v11s" set CONFIG=config_cs2_v11s.ini
if /i "%VARIANT%"=="v11m" set CONFIG=config_cs2_v11m.ini
if /i "%VARIANT%"=="v26m" set CONFIG=config_cs2_v26m.ini
if /i "%VARIANT%"=="v26l" set CONFIG=config_cs2_v26l.ini

if "%CONFIG%"=="" (
    echo Usage: run_cs2.bat [v8^|v11s^|v11m^|v26m^|v26l]
    echo.
    echo   v8    YOLOv8 nano  ^(yolov8n_cs2_20260610.onnx^)
    echo   v11s  YOLOv11 small ^(yolo11s_cs2_20260612.onnx^)
    echo   v11m  YOLOv11 medium ^(yolo11m_cs2_20260612.onnx^)
    echo   v26m  YOLOv26 medium ^(yolo26m_cs2_20260612.onnx^)
    echo   v26l  YOLOv26 large  ^(yolo26l_cs2_20260612.onnx^)
    exit /b 1
)

cd /d "C:\Users\tanguy\Documents\GitHub\YOLOv8-TensorRT-CPP\build2\bin\Release"

echo ========================================
echo   YOLO-TensorRT — CS2
echo   Variant: %VARIANT%
echo   Config:  %CONFIG%
echo ========================================
echo Press Ctrl+C in this window to stop.
echo.

detect_object_image.exe %CONFIG%
