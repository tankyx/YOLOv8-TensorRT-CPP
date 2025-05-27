# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++ implementation of YOLOv8 object detection using NVIDIA TensorRT for optimized inference on gaming applications (CS2/Valorant). The project uses multi-threaded architecture with screen capture, GPU inference, overlay rendering, and automated mouse control.

## Build Commands

```bash
# Configure build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

# Install binaries
cmake --install .
```

## Key Architecture

### Execution Flow
1. **Initialization**: Load INI config → Initialize TensorRT engine → Setup DXGI capture → Connect HID mouse
2. **Capture Thread**: DXGI Desktop Duplication → Double-buffered cv::Mat → `FrameQueue` (drops old frames)
3. **Detection Thread**: 
   - Pop latest frame → Extract ROI → Optional crosshair detection
   - YOLOv8 pipeline: Preprocess (GPU) → TensorRT inference → NMS post-processing
   - Mouse control: Find closest target → Calculate delta → Send HID reports → Auto-click
   - Push to `DetectionQueue` (filters by movement threshold)
4. **Overlay Thread**: D3D11 overlay with detection boxes and metrics (optional)

### Performance Optimizations

#### GPU Acceleration
- TensorRT with FP16/FP32 precision options
- Custom CUDA kernels for FP32→FP16 conversion with 32-byte alignment
- GPU-based preprocessing (resize, BGR→RGB, normalization)
- Batch processing support for higher throughput

#### Memory Management
- **Pre-allocated buffers**: Frame pool, GPU memory allocation at startup
- **Zero-copy operations**: DXGI direct capture, GPU Mat operations
- **Double-buffering**: Avoids allocation during capture
- **Move semantics**: Efficient queue operations without copying

#### Threading & Latency
- **Lock-free design**: Minimal synchronization points
- **Frame dropping**: Keeps only latest frames in queues
- **Thread affinity**: Optional CPU core pinning
- **Direct hardware access**: DXGI for capture, HID for mouse control

### Queue Architecture
- `SafeQueue<T>`: Generic thread-safe queue with condition variables
- `FrameQueue`: Specialized for frames, implements frame dropping
- `DetectionQueue`: Filters detections by movement threshold
- `LatencyQueue`: Sliding window for performance metrics

### Engine Hierarchy
- `EngineBase` → Abstract interface for inference engines
- `EngineFP32` → Standard FP32 precision implementation
- `EngineFP16` → Optimized FP16 with custom CUDA kernels
- `EngineFactory` → Creates engines based on precision type

### Core Components
- `yolov8.cpp/h`: Main inference wrapper with preprocessing/postprocessing
- `DXGICapture.h`: Low-latency screen capture using Desktop Duplication API
- `D3D11Overlay.h`: Hardware-accelerated overlay rendering
- `MouseController.cpp/h`: Direct HID communication for mouse control
- `EngineFP16Kernels.cu`: Optimized CUDA kernels (FP32→FP16 conversion)
- `SoftwareFuser.cu`: Frame fusion for capture card + desktop scenarios
- `threadsafe_queue.h`: Thread-safe queue implementations with move semantics

## Development Guidelines

### Adding New Features
- New engine types should inherit from `EngineBase`
- CUDA kernels go in separate `.cu` files with corresponding headers
- Use the existing threading model and queue system for new processing stages

### Model Conversion
```bash
# Convert PyTorch model to ONNX
python scripts/pytorch2onnx.py model.pt v8n output.onnx [--fp16]
```

### Configuration
Configuration files (INI format) in `dep/` control:
- Model path and precision
- Capture region and scaling
- Mouse sensitivity and automation parameters
- Thread CPU affinity
- Class labels for detection

### Dependencies
- CUDA 12.4+ with compute capability 8.9 (RTX 40 series)
- TensorRT 10.x
- OpenCV with CUDA support
- Windows SDK for DirectX integration

## Important Notes

- The project is Windows-specific due to DirectX dependencies (DXGI, D3D11)
- Thread pinning is used for performance optimization (configurable in INI)
- FP16 precision requires custom CUDA kernels with 32-byte memory alignment
- The software fusion module (`SoftwareFuser.cu`) is for capture card + desktop fusion, not INT8
- Achieves ultra-low latency through:
  - Lock-free queue design with frame dropping
  - Zero-copy DXGI capture
  - Direct HID mouse control (bypasses Windows input system)
  - GPU-accelerated entire pipeline

## Performance Characteristics

- **Capture**: Up to 240 FPS with DXGI Desktop Duplication
- **Inference**: Sub-10ms with FP16 on RTX 40 series
- **End-to-end latency**: < 20ms from screen to mouse action
- **Memory usage**: Pre-allocated pools, no runtime allocations
- **CPU usage**: Minimal due to GPU offloading and efficient threading