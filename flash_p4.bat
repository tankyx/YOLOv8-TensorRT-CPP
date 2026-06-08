@echo off
call "%USERPROFILE%\esp\esp-idf-v5.4.4\export.bat"
cd /d "C:\Users\tanguy\Documents\GitHub\YOLOv8-TensorRT-CPP\ESP32-P4-Aimer"
idf.py -p COM12 flash
