@echo off
setlocal enabledelayedexpansion
cd /d "C:\Users\tanguy\Documents\GitHub\YOLOv8-TensorRT-CPP"

set MODELS=v8n 11n 26n
set GAMES=valorant cs2

for %%G in (%GAMES%) do (
    for %%M in (%MODELS%) do (
        echo ============================================
        echo  TRAINING: %%G - yolo%%M
        echo ============================================
        yolo train model=yolo%%M.pt data=models/%%G/data.yaml epochs=100 imgsz=640
        if !ERRORLEVEL! neq 0 goto :error

        echo Exporting to ONNX...
        yolo export model=runs/detect/train/weights/best.pt format=onnx
        if !ERRORLEVEL! neq 0 goto :error

        move /Y runs\detect\train\weights\best.pt  dep\yolo%%M_%%G.pt
        move /Y runs\detect\train\weights\best.onnx dep\yolo%%M_%%G.onnx
        echo Saved: dep\yolo%%M_%%G.pt / dep\yolo%%M_%%G.onnx

        rmdir /s /q runs\detect\train
        echo.
    )
)

echo ============================================
echo  DONE - Models in dep/
echo ============================================
goto :end

:error
echo TRAINING FAILED with error !ERRORLEVEL!
:end
