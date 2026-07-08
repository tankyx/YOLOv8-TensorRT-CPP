@echo off
cd /d "C:\Users\tanguy\Documents\GitHub\YOLOv8-TensorRT-CPP"
git add dep/config_valo_fp16.ini src/GdiDebugWindow.h src/object_detection_image_mt.cpp
git commit -m "GDI debug window: native Win32 renderer for detections, BGRA preprocess fix, Valorant GPU capture config"
git push
