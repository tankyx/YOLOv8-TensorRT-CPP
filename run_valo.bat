@echo off
cd /d "C:\Users\tanguy\Documents\GitHub\YOLOv8-TensorRT-CPP\build2\bin\Release"
echo Starting YOLOv8-TensorRT for Valorant...
echo Config: config_valo.ini
echo Supports: YOLOv8, YOLOv11, YOLOv26 (auto-detect)
echo Precision: edit config_valo.ini (half/float)
echo Press Ctrl+C in this window to stop.
echo.
detect_object_image.exe config_valo.ini
