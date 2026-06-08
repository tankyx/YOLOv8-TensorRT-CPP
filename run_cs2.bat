@echo off
cd /d "C:\Users\tanguy\Documents\GitHub\YOLOv8-TensorRT-CPP\build2\bin\Release"
echo Starting YOLOv8-TensorRT for CS2...
echo Config: config_cs2.ini
echo Crosshair tracking: ENABLED
echo Debug overlay: ENABLED
echo.
echo Press Ctrl+C in this window to stop.
echo.
detect_object_image.exe config_cs2.ini
