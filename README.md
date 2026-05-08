# YOLOv8-TensorRT-CPP

Sub-millisecond YOLOv8 object detection for real-time gaming aim assist on Windows.
End-to-end pipeline from screen capture to USB HID mouse report stays on the GPU,
inference runs as a captured CUDA graph, and the detection thread does sub-100us
work per frame on an RTX 40-class card.

The detector is a fine-tuned YOLOv8n exported to ONNX, compiled to a TensorRT
engine on first run, and served by a multi-threaded C++ pipeline that does
DXGI Desktop Duplication capture, GPU-side preprocess + decode, and direct USB-HID
mouse output via a small bridge MCU.

## Highlights

- **Fully GPU pipeline** when `UseDirectGpuCapture=true`: DXGI Desktop Duplication
  -> CUDA-D3D11 interop -> fused BGRA->fp16 CHW preproc kernel -> TensorRT
  inference -> GPU anchor filter+decode -> ~20 KB D2H of survivors -> CPU NMS.
  Zero CPU pixel touch.
- **CUDA graph capture** of the entire per-frame GPU sequence. One
  `cudaMemcpy2DAsync` + one `cudaGraphLaunch` per frame instead of seven kernel
  launches.
- **Ultra-low latency**: <0.1 ms detection time on RTX 40 series at 640x640 FP16,
  capture-thread limited; aim-loop latency dominated by display refresh and HID
  bridge round-trip.
- **Direct hardware HID output** to an ESP32-P4 / RP2040 bridge MCU (USB
  composite device) — bypasses Windows raw input entirely, so the game sees a
  hardware mouse.
- **Discord-overlay debug renderer**: detection boxes draw into Discord's
  legacy in-game overlay shared memory (no separate transparent window).
- **Bezier-smoothed dt-based aim path** ported from CS2Miam, plus an unsmoothed
  RMB debug snap-aim for tuning.

## Architecture overview

```
                      +-----------------+
                      |  DXGI Output    |
                      |  Duplication    |
                      +--------+--------+
                               |
              +----------------+----------------+
              | UseDirectGpuCapture = true      |
              |   (DXGICaptureCUDA)             |
              v                                 v
   +------------------------+      +------------------------+
   | D3D11 GPU staging tex  |      | CPU staging tex (Map)  |
   | cudaGraphicsD3D11      |      | cv::cvtColor BGRA->BGR |
   | RegisterResource       |      | cv::Mat                |
   | -> cudaArray (BGRA)    |      |                        |
   | cudaMemcpy2DFromArray  |      |                        |
   |   into BGRA GpuMat     |      |                        |
   +-----------+------------+      +-----------+------------+
               |                               |
               v                               v
       GpuFrameQueue                    FrameQueue (cv::Mat)
               |                               |
               v                               v
   +------------------------+      +------------------------+
   | detectionThreadGpu     |      | detectionThread        |
   | ROI = frame(rect)      |      | ROI = frame(rect)      |
   | (GpuMat view, no copy) |      | + optional crosshair   |
   |                        |      |   GPU template match   |
   +-----------+------------+      +-----------+------------+
               |                               |
               +---------------+---------------+
                               |
                               v
                +------------------------------+
                |   YoloV8::detectObjects      |
                |                              |
                |  cudaMemcpy2D src -> stable  |
                |    capture buffer            |
                |                              |
                |  cudaGraphLaunch:            |
                |    fused preproc kernel      |
                |    (BGR/BGRA -> fp16 CHW)    |
                |    enqueueV3 (TensorRT)      |
                |    memset(survivor count)    |
                |    filter+decode kernel      |
                |    memcpy(count, survivors)  |
                |                              |
                |  cudaStreamSynchronize       |
                |  cv::dnn::NMSBoxesBatched    |
                |    on <= 1024 survivors      |
                +---------------+--------------+
                                |
                                v
                +------------------------------+
                |   MouseController::aim       |
                |                              |
                |  Bezier dt-based smoothing   |
                |    (LMB held)                |
                |   OR unsmoothed snap         |
                |    (RMB held, debug)         |
                |  sub-pixel residue carry     |
                |  HID report -> bridge MCU    |
                +---------------+--------------+
                                |
                                v
                +------------------------------+
                | Discord legacy overlay       |
                | (shared memory, optional)    |
                +------------------------------+
```

### Stages in detail

#### Capture

Two paths, gated by `UseDirectGpuCapture` in the INI:

- **Direct GPU (`true`, recommended).** `DXGICaptureCUDA` creates a private
  GPU-only D3D11 staging texture (`DXGI_FORMAT_B8G8R8A8_UNORM`,
  `D3D11_USAGE_DEFAULT`), registers it once with CUDA via
  `cudaGraphicsD3D11RegisterResource`. Per frame: `AcquireNextFrame` ->
  `CopyResource` desktop into staging -> `cudaGraphicsMapResources` ->
  `cudaGraphicsSubResourceGetMappedArray` -> `cudaMemcpy2DFromArrayAsync` into
  a `cv::cuda::GpuMat` (CV_8UC4 BGRA). No CPU touch, no `cvtColor`, no upload.
  Pushed through `GpuFrameQueue` to the detection thread. The fused preproc
  kernel handles BGRA by reading 4 bytes per pixel and ignoring alpha.

- **Legacy CPU path (`false`).** `DXGICapture` does the classic GPU->CPU staging
  map + `cv::cvtColor` BGRA->BGR + `cv::Mat` upload in the detection thread.
  Slower (1-3 ms per frame just for capture), but compatible with the optional
  `TrackCrosshair` CPU template-matching path.

The two paths are mutually exclusive; `TrackCrosshair=true` requires
`UseDirectGpuCapture=false`. The Discord debug overlay works with either.

#### Preprocess (`EngineFP16Kernels.cu`)

A single CUDA kernel per frame, `fusedPreprocBGRtoFP16Kernel`. One thread per
output pixel:

1. Letterbox resize with bilinear sampling (matches OpenCV's `INTER_LINEAR`
   corner convention; padding region writes zeros to match the original
   `resizeKeepAspectRatioPadRightBottom`).
2. BGR/BGRA -> RGB channel swap (alpha read and dropped on the BGRA path).
3. Per-channel normalize: `(v / 255 - sub[c]) / div[c]` in float.
4. Cast to `__half` and store directly into the FP16 CHW input buffer that's
   bound to TensorRT's input tensor.

Replaces what used to be 7+ OpenCV CUDA launches with a 4.7 MB FP32 intermediate
and an in-preprocess `cudaStreamSynchronize`. The FP16 input buffer
(`m_ownedInputBuffer` in `EngineFP16`) is owned by the engine, allocated once
in `loadNetwork`.

#### Inference (`EngineFP16` + TensorRT 10.x)

Standard `enqueueV3` against the loaded engine. The serialized `.engine` plan
is keyed in `serializeEngineOptions` by ONNX file path, GPU device name,
precision suffix, opt/max batch sizes, and an FNV-1a 64-bit hash of the ONNX
file contents (so editing the `.onnx` in place invalidates the cache).

Output is FP16. There is no host D2H of the full output tensor — see
postprocess.

#### Postprocess (`YoloPostprocKernels.cu`)

A second CUDA kernel runs on the engine's stream right after `enqueueV3`:
`yoloFilterAndDecodeKernel`, one thread per anchor (8400 for 640x640). Each
thread:

1. Loads the anchor's class scores from FP16 output, finds best class.
2. Skips if best score <= `probabilityThreshold`.
3. Decodes xywh -> xyxy in source-image coordinates with ratio scaling and
   clamping to image bounds.
4. `atomicAdd` into a single uint32 counter to claim a slot in the survivor
   buffer, writes `(x0, y0, x1, y1, score, label)` if the slot is < `kMaxSurvivors`
   (1024).

D2H is then ~24 KB (count + survivor records) instead of the ~270 KB FP32 full
output the legacy path moved. CPU does only `cv::dnn::NMSBoxesBatched` on the
survivors plus the top-K cap.

#### CUDA graph capture (`yolov8.cpp`)

The per-frame GPU sequence is `[fused_preproc -> enqueueV3 -> memset(count) ->
filter+decode -> memcpy(count) -> memcpy(survivors)]`. All have stable pointers
because the fused preproc reads from a stable owned `m_captureBuffer` (the
captured `bgr` GpuMat is `cudaMemcpy2DAsync`'d into it before the graph
launch), and the engine's input/output buffers were allocated once at load.

On the first detection-thread call after engine load:

1. **Warmup pass** (outside capture): TRT may make internal allocations the
   first time `enqueueV3` runs; these can't happen inside `cudaStreamCapture`.
2. **Capture pass**: `cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)`,
   re-issue the same sequence, `cudaStreamEndCapture`, `cudaGraphInstantiate`.
3. **Steady state**: `cudaGraphLaunch` per frame.

The graph is rebuilt if capture dimensions change at runtime (they shouldn't,
since `CaptureWidth`/`Height` are INI-fixed).

#### Mouse aim (`MouseController.cpp`)

`MouseController::aim` is called per detection frame. Trigger gating:

- **LMB held** -> smoothed aim. The aim moves toward the closest detection
  using a 2D Bezier curve (`BezierCurve2D`) with a dt-based progress
  parameter: `_t += (1 - smoothingFactor) * 0.00005 * elapsed_us`. This makes
  total time-to-target roughly invariant to detection rate.
- **RMB held** + `DebugAimEnabled=true` -> unsmoothed snap-aim. Raw delta
  multiplied by `DebugSnapGain` and sent in one HID report. For tuning
  in-game sensitivity calibration.
- **LSHIFT held** -> triggerbot hold key. When the crosshair is within a
  detection box and LSHIFT is held, fires a non-blocking randomized-cooldown
  click via the same HID path.

Sub-pixel residue (`_residX/_residY`) is carried frame-to-frame so slow
tracking doesn't truncate to zero counts every report.

The `Smoothing` knob (1-10 in the INI) maps to `smoothingFactor` in
`[0.005, 0.5]`. The mapping was widened from the original CS2Miam 90-100 band
to compensate for the previous pipeline's ~4-5 ms detection staleness; with
the new sub-100us pipeline you generally want `Smoothing=10` (slowest).

#### HID output

`MouseController` opens the bridge MCU as a HID device using
`HidVendorId`/`HidProductId`/`HidSerial` from the INI. Defaults match an
ESP32-P4 board cloned as the `OP1 8K V2` (VID `0x3367` PID `0x1978`, no
serial). Movement reports are 4 bytes: `(buttons, dx_lo, dx_hi, dy_lo, dy_hi)`
with int16 deltas — the bridge re-emits them as native USB HID mouse reports
to the host. Game sees a hardware mouse, not a SendInput synthesized one.

#### Debug overlay (`DiscordOverlay.h`)

When `DebugView=true`, the detection thread renders detection boxes and the
resolved crosshair into a CPU-side Direct2D + WIC bitmap, memcpys the dirty
rect into Discord's legacy in-game-overlay shared memory, and bumps the
frame counter. Discord's already-injected DLL composites our pixels into the
game's swap chain — so the overlay rides the game's frame, no separate
top-most window. Requires the **legacy** Discord in-game overlay (User
Settings -> Game Overlay -> Enable in-game overlay); the post-2024 overlay
doesn't expose this mapping.

`DebugOverlayTargetProcess` (default `cs2.exe`) is the process Discord must
be hooked into. Launch the game first, then this binary.

### Threading & queues

Two threads, plus the main thread:

- `captureThread{,Gpu}` — calls `DXGICapture::CaptureScreen` (or
  `DXGICaptureCUDA::CaptureScreen` with a private CUDA stream and
  `cudaStreamSynchronize` before push), pushes to `FrameQueue` /
  `GpuFrameQueue`. Both queues are drop-old-keep-newest: the consumer always
  sees the most recent frame, never a stale one.
- `detectionThread{,Gpu}` — pops a frame, crops to ROI, runs
  `yoloV8->detectObjects`, calls `mouseController->aim(detections)` and
  `triggerLeftClickIfCenterWithinDetection`, pushes to a `DetectionQueue`
  (move-threshold-filtered), and optionally publishes to the Discord overlay.
- Main thread — pumps Win32 messages, watches for VK_INSERT to exit, prints a
  status line every 100 ms with rolling averages (`Capture / Detection /
  Render`).

`SafeQueue<T>` is a generic mutex+cv FIFO used for the log channel.
`LatencyQueue` keeps a sliding 500 ms window for the rolling averages.

## Prerequisites

### System

- Windows 10/11 (64-bit). DXGI Desktop Duplication + CUDA-D3D11 interop are
  Windows-only.
- NVIDIA GPU with compute capability >= 8.9 (RTX 40 series). RTX 50 series
  also supported with CUDA 12.9; pass `-DCMAKE_CUDA_ARCHITECTURES=120` (or
  whatever your card's SM is) at configure time.
- Visual Studio 2022 with C++ desktop workload + Windows 10 SDK.
- CMake 3.22+.

### CUDA + TensorRT

1. **CUDA Toolkit 12.4+** (12.9 tested) from
   <https://developer.nvidia.com/cuda-downloads>. Verify with `nvcc --version`.
2. **TensorRT 10.x** from <https://developer.nvidia.com/tensorrt>. Extract into
   the CUDA install root, e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\`,
   so the following exist:
   - `include\NvInfer.h`, `include\NvOnnxParser.h`
   - `lib\nvinfer_10.lib`, `lib\nvonnxparser_10.lib`,
     `lib\nvinfer_plugin_10.lib`
   - `bin\nvinfer*.dll` (CMake copies these next to the exe at build time)

### OpenCV with CUDA

Stock OpenCV doesn't ship CUDA modules; build it yourself once.

```cmd
git clone https://github.com/opencv/opencv.git -b 4.10.0
git clone https://github.com/opencv/opencv_contrib.git -b 4.10.0

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

cmake --build . --config Release --target INSTALL
```

For RTX 50 builds, set `-DCUDA_ARCH_BIN=12.0` (or list multiple, e.g. `8.9;12.0`).

### Environment

```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set PATH=%CUDA_PATH%\bin;%PATH%
set OpenCV_DIR=C:\opencv-4.10.0
set PATH=%OpenCV_DIR%\x64\vc17\bin;%PATH%
```

## Building

```cmd
git clone https://github.com/tankyx/YOLOv8-TensorRT-CPP.git
cd YOLOv8-TensorRT-CPP

mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DOpenCV_DIR=C:/opencv-4.10.0 ^
  -DTENSORRT_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"

cmake --build . --config Release
```

Outputs land in `build/bin/Release/`. CMake's POST_BUILD step copies the
contents of `dep/` (model files, INI configs) and TensorRT DLLs alongside the
exe — note that this **overwrites** any in-place edits to the deployed config
on every rebuild. Edit `dep/config_*.ini` if you want changes to survive.

## Model preparation

Convert a YOLOv8 PyTorch checkpoint to ONNX (one-time per model):

```cmd
python scripts/pytorch2onnx.py path\to\your_model.pt v8n dep/yolov8n_cs2_fp16.onnx --fp16
```

Drop the resulting `.onnx` into `dep/` (or wherever `ModelPath` in the INI
points). On first run, `EngineFP16` parses the ONNX and serializes a TensorRT
plan keyed on the file's FNV-1a hash; subsequent launches just deserialize the
cached plan. Re-saving the ONNX invalidates the cache automatically.

## Configuration

Configs live in `dep/`. The shipped ones are:

- `dep/config_cs2.ini` — CS2 (640x640, FP16, 4-class)
- `dep/config_valo_fp16.ini` — Valorant (320x320, FP16)
- `dep/config_valo_fp32.ini` — Valorant (320x320, FP32, fallback)

### Reference (CS2 example, annotated)

```ini
# Model
ModelPath = yolov8n_cs2_fp16.onnx     # relative to working dir
Precision = half                       # half (FP16) or float (FP32)
Labels = c, ch, t, th                  # class names; size determines numClasses

# Capture / ROI
CaptureWidth  = 640                    # ROI fed to the model (must match input)
CaptureHeight = 640
CaptureFPS    = 240                    # advisory; capture is duplication-rate-limited
UseDirectGpuCapture = true             # DXGI/CUDA interop path; recommended

# Detection
HeadLabelID1 = 1                       # primary head class id (used by mouseController)
HeadLabelID2 = 3                       # secondary head class id

# Mouse / aim
CPI              = 3000                # mouse counts per inch (your in-game sens calibration)
MouseSensitivity = 0.80
AimFOV           = 55                  # gates which detections aim engages on
MinGain          = 0.25
MaxGain          = 0.65
MaxSpeed         = 15
Smoothing        = 10                  # 1=snappy, 10=very smooth (Bezier dt-based)
DebugSnapGain    = 3.0                 # RMB snap-aim ROI-px -> mouse-counts multiplier
DebugAimEnabled  = false               # true => RMB engages unsmoothed snap-aim

# HID bridge
HidVendorId  = 0x3367                  # ESP32-P4 OP1 8K V2 default
HidProductId = 0x1978
HidSerial    =                         # blank = match by VID/PID only

# Debug overlay
DebugView                  = false     # Discord legacy overlay debug renderer
DebugOverlayTargetProcess  = cs2.exe   # process Discord must be hooked into

# Optional: CPU template-matching crosshair tracker (legacy capture path only)
TrackCrosshair    = false
CrosshairTemplate = crosshair.png

# Threading (advisory)
PinThreads        = false
CaptureThreadCore = 0
```

### Triggers

- **LMB hold** — smoothed aim engages, HID click bit forwarded to the game.
- **RMB hold** (when `DebugAimEnabled=true`) — unsmoothed snap-aim, click bit
  stays 0 (debug only, no shot fired).
- **LSHIFT hold** — triggerbot. When held and the screen center is inside a
  detection rectangle, fires a non-blocking randomized-cooldown click via HID.

## Running

```cmd
cd build\bin\Release
detect_object_image.exe config_cs2.ini
```

Press **Insert** to exit. The status line shows rolling 500-ms averages of
capture / detection / render latency.

First launch with a new model takes 30-60 seconds to compile the TensorRT
plan; subsequent launches load the cached `.engine` immediately. The first
detection-thread frame after engine load also pays a one-shot CUDA-graph
capture cost (~50 ms); steady-state begins on frame 2.

## Performance

Numbers from an RTX 40-class card at 640x640 FP16, single-class CS2 model:

- **Detection thread time** (preproc + inference + postprocess + sync):
  **<0.1 ms** with `UseDirectGpuCapture=true`. The detection thread is now
  capture-rate-limited.
- **Capture thread time**: dominated by `IDXGIOutputDuplication::AcquireNextFrame`
  blocking until the next refresh; <1 ms of actual work per frame on the
  direct-GPU path, ~3-5 ms on the legacy CPU path.
- **End-to-end latency** (screen change to HID report leaving the host):
  bounded below by display refresh interval + bridge MCU round-trip; the
  software pipeline contributes <2 ms.

Tuning levers, in order of impact:

1. `UseDirectGpuCapture=true` — biggest single jump. Off only if you need
   `TrackCrosshair`.
2. `Precision=half` — keep at FP16. FP32 is for ONNX validation, not
   production.
3. `CaptureWidth/Height` matched to the model input — avoids the inverse
   letterbox math doing real work. The fused preproc kernel handles arbitrary
   sizes but is fastest at 1:1.
4. `Smoothing` — purely an aim-feel knob, doesn't affect detection latency.
   Higher = smoother flick, longer time-to-target.
5. `PinThreads=true` plus `CaptureThreadCore` / `DetectionThreadCore` — only
   useful on systems with bad scheduler behavior; typically negligible.

## Troubleshooting

- **TensorRT libraries not found at link time** — `TENSORRT_ROOT` doesn't
  point at a directory containing `lib\nvinfer_10.lib`. Check the path.
- **`cudaD3D11GetDevice failed` on startup with `UseDirectGpuCapture=true`** —
  CUDA and D3D11 picked different adapters (multi-GPU systems). Set the
  `CUDA_VISIBLE_DEVICES` env var to the adapter that drives your monitor, or
  fall back to `UseDirectGpuCapture=false`.
- **`cudaGraphInstantiate failed`** — extremely rare; usually means a TRT
  internal allocation slipped into the capture. The warmup pass should
  prevent this. If it persists, post the exact CUDA error string from the
  exception.
- **Boxes drawn at the wrong screen position via Discord overlay** — the
  overlay assumes fullscreen at native resolution. Windowed/borderless or
  scaled-resolution will draw boxes at the wrong screen location.
- **Discord overlay shows nothing** — must be the *legacy* in-game overlay
  enabled in Discord settings; the post-2024 overlay doesn't expose the
  shared-memory mapping. Discord must already be hooked into the target
  process when the detector starts (launch the game first).
- **HID device not opening** — `HidVendorId` / `HidProductId` in the INI
  don't match a connected device. Check Device Manager for the bridge MCU's
  VID/PID. Run as Administrator if device enumeration fails.
- **DXGI access lost** — happens on resolution changes, lock screen, UAC.
  Both capture paths recover on the next frame (re-create the duplication +
  re-register the CUDA resource).

## Repository layout

```
src/
  object_detection_image_mt.cpp   # main, ObjectDetectionSystem, threads
  yolov8.{h,cpp}                  # detector wrapper (engine + GPU postproc + graph)
  EngineBase.h                    # virtual interface, shared engine helpers
  EngineFP16.h                    # FP16 engine (the production path)
  EngineFP32.h                    # FP32 engine (legacy fallback)
  EngineFactory.h                 # selects engine by Precision
  EngineFP16Kernels.{h,cu}        # fused BGR/BGRA -> fp16 CHW preproc kernel
  YoloPostprocKernels.{h,cu}      # GPU anchor filter + decode kernel
  DXGICapture.h                   # legacy CPU capture path
  DXGICaptureCUDA.h               # direct GPU/D3D11 interop capture path
  DiscordOverlay.h                # Discord legacy overlay debug renderer
  MouseController.{h,cpp}         # HID output, Bezier aim, triggerbot
  threadsafe_queue.h              # SafeQueue, FrameQueue, GpuFrameQueue, etc.
  IniParser.h                     # zero-dep INI parser
  SoftwareFuser.{h,cu}            # capture-card + desktop fusion (optional)

dep/
  *.ini                           # configs per game
  *.onnx                          # exported model weights
  *.engine                        # cached TensorRT plans (generated)

scripts/
  pytorch2onnx.py                 # PyTorch -> ONNX export
```

## Acknowledgments

- Ultralytics — YOLOv8.
- NVIDIA — TensorRT, CUDA, cuda-d3d11 interop.
- OpenCV — CUDA imgproc + DNN NMS.
- CS2Miam — original Bezier smoothing pipeline that the aim path was ported
  from.
- Samuel Tulach (OverlayCord) — reverse-engineered the Discord legacy
  in-game-overlay shared-memory contract used by `DiscordOverlay`.

## License

MIT — see `LICENSE`.
