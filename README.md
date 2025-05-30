# YOLOv8-TensorRT-CPP

Ultra-low latency YOLOv8 object detection using NVIDIA TensorRT for real-time gaming applications. Achieves sub-20ms end-to-end latency from screen capture to mouse action.

## Features

- **High Performance**: Sub-10ms inference with FP16 precision on RTX 40 series
- **Low Latency**: < 20ms end-to-end latency using lock-free design
- **GPU Accelerated**: Entire pipeline runs on GPU (capture, inference, post-processing)
- **Multi-threaded**: Separate threads for capture, detection, and overlay
- **Direct Hardware Access**: DXGI screen capture and HID mouse control
- **Flexible Precision**: Support for both FP32 and FP16 inference

## Prerequisites

### System Requirements
- Windows 10/11 (64-bit)
- NVIDIA GPU with compute capability 8.9+ (RTX 40 series recommended)
- Visual Studio 2022 with C++ development tools
- CMake 3.22 or higher

### Required Software

1. **CUDA Toolkit 12.4**
   - Download from: https://developer.nvidia.com/cuda-12-4-0-download-archive
   - Install with default settings
   - Verify installation: `nvcc --version`

2. **TensorRT 10.x**
   - Download from: https://developer.nvidia.com/tensorrt-download
   - Extract to CUDA installation directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`)
   - The following files should be in your CUDA directory:
     - `include\NvInfer.h`
     - `lib\nvinfer_10.lib`
     - `lib\nvonnxparser_10.lib`
     - `lib\nvinfer_plugin_10.lib`

3. **Windows SDK**
   - Install via Visual Studio Installer
   - Required for DirectX development (DXGI, D3D11)

## Building OpenCV with CUDA Support

### 1. Download OpenCV
```bash
git clone https://github.com/opencv/opencv.git -b 4.10.0
git clone https://github.com/opencv/opencv_contrib.git -b 4.10.0
```

### 2. Configure OpenCV with CMake
```bash
mkdir opencv-build && cd opencv-build

cmake -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX=C:/opencv-4.10.0 ^
  -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ^
  -DWITH_CUDA=ON ^
  -DWITH_CUDNN=ON ^
  -DOPENCV_DNN_CUDA=ON ^
  -DCUDA_ARCH_BIN=8.9 ^
  -DCUDA_FAST_MATH=ON ^
  -DWITH_CUBLAS=ON ^
  -DBUILD_opencv_world=ON ^
  -DBUILD_TESTS=OFF ^
  -DBUILD_PERF_TESTS=OFF ^
  -DBUILD_EXAMPLES=OFF ^
  ../opencv
```

### 3. Build and Install OpenCV
```bash
cmake --build . --config Release --target INSTALL
```

## Environment Variables

Set the following environment variables before building:

```bash
# CUDA paths
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_PATH%\bin;%PATH%

# OpenCV paths
set OpenCV_DIR=C:\opencv-4.10.0
set PATH=%OpenCV_DIR%\x64\vc17\bin;%PATH%

# TensorRT paths (if installed separately)
set TENSORRT_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
```

## Building the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/YOLOv8-TensorRT-CPP.git
cd YOLOv8-TensorRT-CPP
```

### 2. Configure with CMake
```bash
mkdir build && cd build

cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DOpenCV_DIR=C:/opencv-4.10.0 ^
  -DTENSORRT_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
```

### 3. Build the Project
```bash
cmake --build . --config Release
```

### 4. Install (Optional)
```bash
cmake --install .
```

## Configuration

### Model Preparation

1. **Convert PyTorch model to ONNX**:
```bash
python scripts/pytorch2onnx.py your_model.pt v8n output.onnx --fp16
```

2. **Place model in dep/ folder**:
```bash
cp output.onnx dep/yolov8n_val_fp16.onnx
```

### Configuration File

Edit the INI configuration file (e.g., `dep/config_valo_fp16.ini`):

```ini
# Model settings
ModelPath = yolov8n_val_fp16.onnx
Precision = half                    # Options: half (FP16) or float (FP32)

# Capture settings
CaptureWidth = 320                  # Capture region width
CaptureHeight = 320                 # Capture region height
CaptureFPS = 240                    # Target capture FPS

# Detection settings
HeadLabelID1 = 1                   # Class ID for head detection
HeadLabelID2 = 1                   # Secondary class ID
Labels = ENEMY_BODY, ENEMY_HEAD    # Class labels

# Mouse control
MouseSensitivity = 0.80            # Mouse movement sensitivity
AimFOV = 50                        # Field of view for targeting
MinGain = 0.25                     # Minimum movement gain
MaxGain = 0.65                     # Maximum movement gain
MaxSpeed = 20                      # Maximum mouse speed
CPI = 3000                         # Mouse CPI/DPI

# Performance settings
PinThreads = true                  # Pin threads to CPU cores
CaptureThreadCore = 0              # CPU core for capture thread
FusionThreadCore = 2               # CPU core for fusion thread

# Features
UseOverlay = false                 # Enable/disable overlay
TrackCrosshair = false            # Track crosshair position
UseFusion = false                 # Use capture card fusion
```

## Running the Application

```bash
cd build/bin/Release
detect_object_image.exe config_valo_fp16.ini
```

### Command Line Options
- Pass configuration file as argument: `detect_object_image.exe config.ini`
- Default config file is used if no argument provided

## Troubleshooting

### Common Issues

1. **TensorRT libraries not found**
   - Ensure TensorRT is installed in CUDA directory
   - Check that version numbers match (e.g., `nvinfer_10.lib`)
   - Verify `TENSORRT_ROOT` environment variable

2. **OpenCV not found**
   - Set `OpenCV_DIR` to OpenCV installation directory
   - Ensure OpenCV was built with CUDA support
   - Add OpenCV bin directory to PATH

3. **CUDA errors**
   - Verify CUDA 12.4 is installed
   - Check GPU compute capability (needs 8.9+)
   - Update GPU drivers to latest version

4. **DirectX errors**
   - Install latest Windows SDK
   - Ensure D3D11 libraries are available
   - Run as administrator if capture fails

### Performance Optimization

1. **Enable FP16 precision**: Use `Precision = half` in config
2. **Adjust capture resolution**: Lower resolution = faster inference
3. **Pin threads to cores**: Set `PinThreads = true`
4. **Disable overlay**: Set `UseOverlay = false` if not needed
5. **Optimize batch size**: Adjust based on GPU memory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- NVIDIA TensorRT for inference optimization
- OpenCV for image processing
- DirectX for screen capture and overlay