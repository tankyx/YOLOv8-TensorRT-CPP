@echo off
cd /d "C:\Users\tanguy\Documents\GitHub\YOLOv8-TensorRT-CPP\build2\bin\Release"
echo Starting YOLOv8-TensorRT for CS2...
echo Config: config_cs2.ini
echo Precision: FP16
echo Supports: YOLOv8, YOLOv11, YOLOv26 (auto-detect)
echo Press Ctrl+C in this window to stop.
echo.
detect_object_image.exe config_cs2.ini
