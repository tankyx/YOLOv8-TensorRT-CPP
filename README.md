# YOLOv8-TensorRT-CPP

Sub-millisecond YOLO object detection for real-time gaming aim assist on Windows.
End-to-end GPU pipeline: DXGI capture → fused preproc → TensorRT inference → GPU
decode → HID mouse output. CUDA graph-captured, Discord-overlay-debugged.

Supports YOLOv8, YOLOv11, and YOLOv26 models.

---

## Quick Start

```cmd
# 1. Build
git clone https://github.com/tankyx/YOLOv8-TensorRT-CPP.git
cd YOLOv8-TensorRT-CPP
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DOpenCV_DIR=C:/opencv-4.10.0 ^
  -DTENSORRT_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"
cmake --build . --config Release

# 2. Export your model to ONNX (one-time)
python scripts/pytorch2onnx.py path\to\model.pt v8n dep/model.onnx --fp16

# 3. Configure
# Edit dep/config_cs2.ini — set ModelPath, Labels, CPI, etc.
# See the annotated reference below.

# 4. Run
cd build\bin\Release
detect_object_image.exe config_cs2.ini
```

First launch with a new model takes 30–60 seconds to compile the TensorRT plan.
Press **Insert** to exit.

Batch scripts at repo root:
```cmd
run_cs2.bat              # CS2 — default YOLOv8n
run_cs2.bat v11s          # YOLOv11s
run_cs2.bat v11m          # YOLOv11m
run_valo.bat              # Valorant — YOLOv8n
run_valo.bat v11s          # YOLOv11s
run_valo.bat v26l          # YOLOv26l
```

Prerequisites: Windows 10/11, NVIDIA RTX 40+ series, Visual Studio 2022, CMake 3.22+,
CUDA Toolkit 12.4+, TensorRT 10.x, OpenCV 4.10 with CUDA modules.
Full build instructions → [Building from source](#building-from-source).

---

## Highlights

- **Fully GPU pipeline** with `UseDirectGpuCapture=true`: DXGI Desktop Duplication
  → CUDA-D3D11 interop → fused BGRA→fp16 CHW preproc kernel → TensorRT
  inference → GPU anchor filter+decode → ~20 KB D2H → CPU NMS. Zero CPU pixel touch.
- **CUDA graph capture** of the per-frame GPU sequence. One `cudaMemcpy2DAsync` +
  one `cudaGraphLaunch` per frame.
- **~0.4 ms detection time** on RTX 5090 at 640×640 FP16 with direct GPU capture.
- **Direct HID output** to an ESP32-P4/RP2040 bridge MCU — the game sees a hardware
  mouse, not a synthesized one.
- **Discord-overlay debug renderer**: detection boxes draw into Discord's legacy
  in-game overlay shared memory (no separate window).
- **FOV-based pixel-perfect aim** — focal-length/atan2 conversion from screen pixels
  to mouse counts, calibrated per-game (m_yaw + rendering hFOV). No CPI dependency.
- **Recoil containment strategy** — aimbot keeps the crosshair inside the target's
  bounding box rather than snapping to centre, absorbing aim-punch naturally.
- **GPU-only pipeline** — CPU capture/detection paths removed. Direct GPU capture
  is the only path.

---

## Running

```cmd
# Direct
cd build\bin\Release
detect_object_image.exe config_valo.ini

# Or use batch scripts
run_cs2.bat v11s
```

Press **Insert** to exit. Status line shows rolling 500 ms averages of
capture / detection / render latency.

### Triggers

| Key held | Behavior |
|---|---|
| **LMB** | Recoil-containment aim — keeps crosshair inside the target bounding box |
| **LSHIFT** | Triggerbot — fires when crosshair is inside a detection box |

---

## Configuration

Configs live in `dep/`. One per game + model variant:

```
dep/
  config_cs2.ini               # CS2 base
  config_cs2_v11s.ini           # CS2 YOLOv11s
  config_valo.ini               # Valorant base
  config_valo_v11m.ini          # Valorant YOLOv11m
  config_valo_v11s_purple.ini   # Valorant purple enemy highlight
  ...
```

### Full INI reference

```ini
# Model
ModelPath    = yolov8n_cs2_fp16.onnx    # relative to working dir
ModelVersion = auto                      # auto, v8, v11, or v26
Precision    = half                      # half (FP16) or float (FP32)
Labels       = player                   # class names

# Capture
CaptureWidth  = 640
CaptureHeight = 640
CaptureFPS    = 240

# Detection
HeadLabelID1  = 0                       # primary head class id
HeadLabelID2  = 0                       # secondary head class id

# Aim — pixel-perfect FOV-based conversion
MouseSensitivity = 0.25                  # in-game sensitivity (CS2) or 0.1 (Valorant)
GameFOV          = 106                   # rendering hFOV (106 for CS2 16:9, 103 Valorant)
AimFOV           = 45                    # aim activation radius in screen pixels
Smoothing        = 4                     # 1=snappy (high gain), 10=very smooth (low gain)
DebugSnapGain    = 1.0                   # multiplier on RMB snap counts (1.0 = pixel-perfect)
DebugAimEnabled  = false                 # RMB snap-aim disabled by default

# HID bridge
HidVendorId  = 0x3367                   # ESP32-P4 OP1 8K V2 default
HidProductId = 0x1978
HidSerial    =                          # blank = match by VID/PID only

# Debug overlay (Discord legacy in-game overlay)
DebugView                 = true
DebugOverlayTargetProcess = cs2.exe

# Threading
PinThreads         = false
CaptureThreadCore  = 0

# Monitoring (optional JSON status file for codewhale)
MetricsStatus = status.json
```

CMake's POST_BUILD step copies `dep/` next to the exe — it **overwrites** deployed
configs on rebuild. Edit `dep/config_*.ini` to persist changes.

---

## Model preparation

Export a YOLO checkpoint to ONNX with `scripts/pytorch2onnx.py`:

```cmd
# YOLOv8 / YOLOv11
python scripts/pytorch2onnx.py path\to\model.pt v8n dep/model.onnx --fp16

# YOLOv26 (NMS-free)
python scripts/pytorch2onnx.py path\to\model.pt v26n dep/model.onnx --fp16
```

Drop the `.onnx` into `dep/` and set `ModelPath` in your INI. The engine auto-detects
the YOLO version from filename and output shape. On first run, TensorRT compiles and
serializes a `.engine` plan keyed on the ONNX file hash — subsequent launches load it
instantly.

Version tokens: `v8n`, `v8s`, `v8m`, `v8l`, `v8x`, `v11n`, `v11s`, `v11m`, `v11l`,
`v11x`, `v26n`, `v26s`, `v26m`, `v26l`, `v26x`.

---

## Building from source

### Prerequisites

- **Windows 10/11 (64-bit)** — DXGI Desktop Duplication + CUDA-D3D11 interop are
  Windows-only.
- **NVIDIA GPU** with compute capability ≥ 8.9 (RTX 40 series+).
- **Visual Studio 2022** with C++ desktop workload + Windows 10 SDK.
- **CMake 3.22+**.
- **CUDA Toolkit 12.4+** (12.9 tested). Install from
  [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
- **TensorRT 10.x** from [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt).
  Extract into the CUDA install root so `include\NvInfer.h`, `lib\nvinfer_10.lib`,
  and `bin\nvinfer*.dll` are resolvable.
- **OpenCV 4.10 with CUDA modules**. Stock OpenCV doesn't ship CUDA — build from source:
  ```cmd
  git clone https://github.com/opencv/opencv.git -b 4.10.0
  git clone https://github.com/opencv/opencv_contrib.git -b 4.10.0
  mkdir opencv-build && cd opencv-build
  cmake -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX=C:/opencv-4.10.0 ^
    -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ^
    -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON ^
    -DCUDA_ARCH_BIN=8.9 -DCUDA_FAST_MATH=ON -DWITH_CUBLAS=ON ^
    -DBUILD_opencv_world=ON ^
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF ^
    ../opencv
  cmake --build . --config Release --target INSTALL
  ```
  For RTX 50 series, use `-DCUDA_ARCH_BIN=12.0`.

### Environment

```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set PATH=%CUDA_PATH%\bin;%PATH%
set OpenCV_DIR=C:\opencv-4.10.0
set PATH=%OpenCV_DIR%\x64\vc17\bin;%PATH%
```

### Build

```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DOpenCV_DIR=C:/opencv-4.10.0 ^
  -DTENSORRT_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"
cmake --build . --config Release
```

Output: `build/bin/Release/detect_object_image.exe`.

---

## Architecture overview

```
                       +-----------------+
                       |  DXGI Output    |
                       |  Duplication    |
                       +--------+--------+
                                |
                                v
                    +-----------------------+
                    | D3D11 GPU staging tex |
                    | cudaGraphicsD3D11     |
                    | → cudaArray (BGRA)   |
                    | → GpuMat (BGRA)       |
                    +-----------+-----------+
                                |
                                v
                        GpuFrameQueue
                                |
                                v
                    +-----------------------+
                    | detectionThread       |
                    | ROI = frame(rect)     |
                    | (GpuMat view, no copy)|
                    +-----------+-----------+
                                |
                                v
                 +------------------------------+
                 |   YoloDetector::detectObjects |
                 |                              |
                 |  cudaMemcpy2D → capture buf  |
                 |  cudaGraphLaunch:            |
                 |    fused preproc kernel      |
                 |    (BGR/BGRA → fp16 CHW)     |
                 |    enqueueV3 (TensorRT)      |
                 |    memset(survivor count)    |
                 |    filter+decode kernel      |
                 |    memcpy(count, survivors)  |
                 |  cudaStreamSynchronize       |
                 |  NMS (V8/V11) or threshold   |
                 |    (V26, NMS-free)           |
                 +---------------+--------------+
                                 |
                                 v
                 +------------------------------+
                 |   MouseController::aim       |
                 |                              |
                 |  recoil containment          |
                 |  (keep crosshair in box)     |
                 |  distance-adaptive gain      |
                 |  FOV pixel→count conversion  |
                 |  sub-pixel residue carry     |
                 |  HID report → bridge MCU     |
                 +---------------+--------------+
```

### Pipeline stages

**Capture.** Single GPU path via CUDA-D3D11 interop — DXGI Desktop Duplication
→ CUDA array → GpuMat. Zero CPU pixel touch.

**Preprocess.** A single fused CUDA kernel (`EngineFP16Kernels.cu`): letterbox
resize with bilinear sampling, BGR/BGRA→RGB channel swap, per-channel normalize,
cast to `__half`. Writes directly into the TensorRT input buffer — no intermediate
FP32 blob, no OpenCV CUDA launches.

**Inference.** TensorRT 10.x `enqueueV3` against the loaded engine. Engine plans
are cached and keyed on the ONNX file hash.

**Postprocess.** A second CUDA kernel runs on the engine's stream after inference:
- **YOLOv8**: xywh→xyxy decode, best class, atomic survivor write.
- **YOLOv11**: DFL softmax over 16 distribution bins per coordinate, then decode.
- **YOLOv26**: end-to-end, confidence threshold only — no box decode, no NMS.

D2H is ~24 KB (count + ≤1024 survivor records). CPU does only NMS (V8/V11) or
direct threshold pass (V26).

**CUDA graph.** The per-frame GPU sequence is captured as a CUDA graph on the
first detection frame after engine load. Steady-state is one `cudaGraphLaunch`
per frame.

**Mouse aim.** `MouseController::aim` uses a recoil-containment strategy: the
aimbot only moves when the crosshair exits the target's bounding box. Pixel
deltas are converted to mouse counts via a focal-length/atan2 FOV method
calibrated per-game (m_yaw + rendering hFOV). A distance-adaptive proportional
gain provides smooth convergence without oscillation. Sub-pixel residue carries
frame-to-frame. HID reports go to an ESP32-P4/RP2040 bridge MCU that re-emits
them as native USB mouse reports.

**Debug overlay.** `DiscordOverlay.h` renders detection boxes into Discord's
legacy in-game overlay shared memory — boxes ride the game's swap chain with
no separate window. Requires the legacy Discord overlay enabled.

### Threading

Three threads: capture (DXGI Duplication → queue), detection (queue → crop →
detect → aim → overlay), and main (Win32 message pump + status line).

Queues use drop-old-keep-newest semantics. `LatencyQueue` maintains a 500 ms
sliding window for the rolling averages.

---

## YOLO model versions

| Version | Output shape | Postprocess | NMS needed? |
|---|---|---|---|
| YOLOv8 | `(1, 4+C, 8400)` | xywh→xyxy decode | Yes |
| YOLOv11 | `(1, 4×16+C, 8400)` | DFL softmax + decode | Yes |
| YOLOv26 | `(1, 300, 6)` | Confidence threshold | No |

Detector auto-detection: `DetectorFactory::create()` checks filename first, then
output shape. Override with `ModelVersion` in the INI.

---

## Performance

Measured on RTX 5090 at 640×640 FP16:

- **Detection thread**: ~0.4 ms (YOLOv8n); ~2 ms (YOLOv26l).
- **Capture thread**: dominated by `AcquireNextFrame` blocking for next refresh.
- **End-to-end latency**: bounded by display refresh + bridge MCU round-trip.

### Tuning

1. `Precision=half` — FP16 is the production path.
2. Match `CaptureWidth/Height` to model input for fastest fused preproc.
3. `GameFOV` — must match your game's actual rendering hFOV (CS2 16:9 = 106, Valorant = 103).
4. `MouseSensitivity` — must match your in-game sensitivity exactly.
5. `Smoothing` — controls the proportional gain (1=snappy high gain, 10=very smooth low gain).

---

## Troubleshooting

- **TensorRT libraries not found at link time** — check `TENSORRT_ROOT` path.
- **`cudaD3D11GetDevice failed`** — CUDA and D3D11 picked different GPUs. Set
  `CUDA_VISIBLE_DEVICES` or use `UseDirectGpuCapture=false`.
- **`cudaGraphInstantiate failed`** — rare; the warmup pass should prevent it.
- **Discord overlay shows nothing** — must use the *legacy* in-game overlay in
  Discord settings. Launch game first, then detector.
- **HID device not opening** — verify VID/PID in Device Manager. Run as Admin if
  enumeration fails.
- **DXGI access lost** — on resolution change / lock screen / UAC. Both capture
  paths auto-recover next frame.

---

## Repository layout

```
src/
  object_detection_image_mt.cpp   # main, ObjectDetectionSystem, GPU-only threads
  YoloDetector.{h,cpp}            # base detector (engine mgmt, graph, GPU buffers)
  yolov8.{h,cpp}                  # YOLOv8 (xywh decode + NMS)
  YoloV11.{h,cpp}                 # YOLOv11 (DFL softmax + NMS)
  YoloV26.{h,cpp}                 # YOLOv26 (end-to-end, NMS-free)
  DetectorFactory.{h,cpp}         # auto-detection from filename + output shape
  EngineBase.h                    # virtual interface, build helpers
  EngineFP16.h                    # FP16 engine (production path)
  EngineFP32.h                    # FP32 engine (fallback)
  EngineFactory.h                 # selects engine by Precision
  EngineFP16Kernels.{h,cu}        # fused BGR/BGRA → fp16 CHW preproc kernel
  YoloPostprocKernels.{h,cu}     # GPU filter+decode kernels (V8/V11/V26)
  DXGICaptureCUDA.h               # GPU/D3D11 interop capture
  DiscordOverlay.h                # Discord legacy overlay debug renderer
  MouseController.{h,cpp}         # HID output, FOV pixel→count, containment aim
  threadsafe_queue.h              # thread-safe queues (frame, detection, latency)
  IniParser.h                     # zero-dep INI parser
  MetricsWriter.h                 # optional JSON status file writer

dep/
  config_{cs2,valo}.ini           # base configs per game
  config_{cs2,valo}_v{8,11s,11m,26m,26l}.ini  # per-variant configs
  *.onnx, *.pt                    # models / checkpoints
  *.engine                        # cached TensorRT plans (generated on first run)

scripts/
  pytorch2onnx.py                 # PyTorch → ONNX export

ESP32-P4-Aimer/                   # Bridge MCU firmware (ESP-IDF project)
```

---

## License

MIT — see `LICENSE`.
