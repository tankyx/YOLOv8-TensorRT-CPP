@echo off
cd /d "C:\Users\tanguy\Documents\GitHub\YOLOv8-TensorRT-CPP\build2\bin\Release"
echo Starting YOLOv8-TensorRT for Valorant (FP32)...
echo Config: config_valo_fp32.ini
echo Debug overlay: ENABLED
echo Precision: FP32
echo.
echo Press Ctrl+C in this window to stop.
echo.
detect_object_image.exe config_valo_fp32.ini
