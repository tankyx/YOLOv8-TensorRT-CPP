@echo off
cd /d "C:\Users\tanguy\Documents\GitHub\YOLOv8-TensorRT-CPP"
git merge --abort 2>nul
git reset --hard HEAD
git pull
