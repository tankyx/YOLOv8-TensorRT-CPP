# Follow-ups

Engineering work deferred from the `cleanup-optim-robustness` branch, plus two adjacent investigations the user requested:

1. [Deferred optimizations](#1-deferred-optimizations)
   - [c25 — fused preprocess CUDA kernel](#c25--fused-preprocess-cuda-kernel)
   - [c26 — CUDA-D3D11 interop capture path](#c26--cuda-d3d11-interop-capture-path)
   - [c28 — overlap crosshair GPU match with TRT inference](#c28--overlap-crosshair-gpu-match-with-trt-inference)
2. [INT8 inference path](#2-int8-inference-path)
3. [Newer YOLO versions](#3-newer-yolo-versions)

Everything below assumes the post-cleanup state of the tree (engines unified under `EngineBase`, persistent `m_stream`, FP16 input/output GPU kernels, no overlay, **GPU crosshair tracking via `cv::cuda::TemplateMatching`** — c27).

---

## 1. Deferred optimizations

c25 and c26 were ranked high-value / high-risk in the original plan and skipped because the host running development is Linux while the project is Windows-only — both touch correctness in ways that desk-review can't catch (sub-pixel detection drift in c25, format/synchronization gotchas in c26). c28 is a small follow-on to c27 (GPU crosshair match) that reclaims its remaining latency by stream-overlapping with TRT inference. Land all three with smoke-tests on Windows, not blind.

### c25 — fused preprocess CUDA kernel

#### Why

Current preprocess at `src/yolov8.cpp:37 YoloV8::preprocess()` and `src/EngineBase.h blobFromGpuMats()` chains six OpenCV CUDA calls:

1. `cv::cuda::cvtColor(BGR → RGB)`
2. `EngineBase::resizeKeepAspectRatioPadRightBottom` (which calls `cv::cuda::resize` + `cv::cuda::GpuMat::copyTo` for the letterbox pad)
3. `cv::cuda::split` to interleave HWC → CHW
4. `cv::cuda::GpuMat::convertTo` (uint8 → float, /255)
5. `cv::cuda::subtract`
6. `cv::cuda::divide`
7. (FP16 only) `launchConvertFP32ToFP16Kernel` from `EngineFP16Kernels.cu`

That's six to seven kernel launches per frame, each with launch overhead and a separate global-memory pass. For a 320×320×3 input the math is trivial — it's launch overhead + memory bandwidth that costs.

Estimated win: 1–2 ms per detection on RTX 40 series, depending on input size.

#### Design

Single kernel `preprocessBgrToBlob`:

- Input: source `cv::cuda::GpuMat` (BGR uint8, arbitrary `srcH × srcW × 3`).
- Output: device pointer to a CHW `float` (or `__half`) blob of shape `1 × 3 × dstH × dstW`.
- Per-thread work: one output pixel at `(c, dy, dx)` in dst space. Compute the inverse-letterbox source `(sy, sx)` in float, sample the source with bilinear interpolation (or nearest, matching current behavior — `cv::cuda::resize` with default `INTER_LINEAR` is bilinear), apply BGR→RGB swap (`channelSrc = 2 - c` for the YOLO RGB channel order), divide by 255, optional subVals/divVals normalization, optional `__float2half` cast.
- Threads outside the un-padded region write the letterbox bgcolor (currently `cv::Scalar(0, 0, 0)` so just zero).

The letterbox math reproduces `EngineBase::resizeKeepAspectRatioPadRightBottom` (`src/EngineBase.h:106`):

```
r = min(dstW / srcW, dstH / srcH)
unpadW = srcW * r,  unpadH = srcH * r
sx = dx / r,        sy = dy / r
out-of-bounds (dx >= unpadW || dy >= unpadH) → write bgcolor
```

#### File-level plan

- New `src/PreprocessKernels.cu` exporting two `extern "C"` launchers:
  - `launchPreprocessBgrToBlobFP32(const uint8_t* src, int srcStep, int srcW, int srcH, float* dst, int dstW, int dstH, float subB, float subG, float subR, float divB, float divG, float divR, bool normalize, cudaStream_t stream)`
  - `launchPreprocessBgrToBlobFP16(...)` — same parameters, `__half* dst`.
- New `src/PreprocessKernels.h` with the C declarations (mirror `EngineFP16Kernels.h`).
- `CMakeLists.txt` — add `PreprocessKernels.cu` to a CUDA static lib (or to the existing `engine_cuda_kernels` target).
- `src/EngineFP32.h::runInference` and `src/EngineFP16.h::runInference` — replace the call to `blobFromGpuMats(...)` (and, for FP16, the FP32→FP16 cast) with a direct call to the new launcher writing into a pre-allocated, persistent device blob owned by the engine. Allocate that blob in `loadNetwork()` once we know `inputDims`.
- `src/yolov8.cpp::preprocess` collapses into "compute `m_ratio`, hand the source GpuMat directly to the engine"; the engine no longer takes a `vector<vector<GpuMat>>` for input — the API becomes `runInference(const cv::cuda::GpuMat& src, vector<vector<vector<float>>>& out)`. Keep the old API as a deprecated overload if you want the migration to be staged.

#### Verification (mandatory before merge)

- Bit-comparable detection coords on a fixed input video vs the post-c24 baseline (FP32 path: bit-exact; FP16 path: ±1 px tolerance — the float-vs-half summation order may drift sub-pixel).
- Visual sanity: render a single frame, overlay the detection boxes, eyeball the alignment vs the OpenCV-CUDA pipeline. A bug in the inverse-letterbox math will shift boxes by one or two pixels uniformly — easy to miss in latency numbers, obvious visually.
- Boundary tests: aspect ratios that produce non-square unpad regions (e.g. 1920×1080 → 640×640 letterbox), tiny inputs, inputs where `srcW == dstW`.

#### Risk register

- **Bilinear sampling vs OpenCV's**: OpenCV's `cv::cuda::resize` with `INTER_LINEAR` follows a specific corner convention. Reproduce it exactly or accept a documented sub-pixel drift.
- **Normalize semantics**: current code does `gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f)` then `cv::cuda::subtract` and `cv::cuda::divide`. The kernel must apply these in the same order `(x/255 - sub) / div` and the project currently uses `subVals = {0,0,0}, divVals = {1,1,1}` so most paths are no-ops — but verify for any future configs.
- **FP16 NaN/Inf**: only an issue if normalize introduces a divide-by-zero. Current divVals are 1, so safe.

---

### c26 — CUDA-D3D11 interop capture path

#### Why

Current capture path in `src/DXGICapture.h::CaptureScreen`:

1. `IDXGIOutputDuplication::AcquireNextFrame` → GPU texture.
2. `ID3D11DeviceContext::CopyResource` → CPU-readable staging texture (GPU → GPU copy).
3. `Map(D3D11_MAP_READ)` → CPU pointer.
4. `cv::cvtColor(BGRA → BGR)` on the CPU.
5. Push `cv::Mat` into `FrameQueue`.
6. Detection thread does `gpuImg.upload(inputImageBGR)` at `src/yolov8.cpp:96` — CPU → GPU.

That's a full GPU → CPU → GPU round trip per frame, plus a CPU `cvtColor`. For 1920×1080 capture that's ~8 MB of pixel traffic each way per frame.

Estimated win: 1–3 ms per frame depending on capture resolution and PCIe topology.

#### Design

Register the duplicated desktop texture (or a dedicated D3D11 staging texture in GPU memory) as a CUDA resource via `cudaGraphicsD3D11RegisterResource`, map it per frame with `cudaGraphicsMapResources`, get a `cudaArray_t` via `cudaGraphicsSubResourceGetMappedArray`, and feed that array directly into the c25 fused preprocess kernel — which is also why c26 should land **after** c25.

This eliminates the CPU mapping, the CPU `cvtColor` (BGRA → RGB happens inside the preprocess kernel), and the upload.

#### File-level plan

- New `src/DXGICaptureCUDA.h` — alternative implementation behind a `UseDirectGpuCapture` flag (defaults `false`). Same public interface as `DXGICapture` plus `cv::cuda::GpuMat captureGpu()` returning a GpuMat view of the mapped texture.
- New `src/threadsafe_queue.h::GpuFrameQueue` — same drop-old-keep-newest semantics as `FrameQueue` but holding `cv::cuda::GpuMat`. The `cv::cuda::GpuMat`'s shallow-copy refcounting works the same as `cv::Mat`'s.
- `src/object_detection_image_mt.cpp::ObjectDetectionSystem::initializeSystem` — read `UseDirectGpuCapture`, branch on it: instantiate either `DXGICapture` + `FrameQueue` or `DXGICaptureCUDA` + `GpuFrameQueue`. Use `std::variant` or two separate threads — the latter is simpler.
- `src/yolov8.h` — add a `detectObjects(const cv::cuda::GpuMat& bgrInput)` overload that skips the upload. Already partially there at `yolov8.h:57` — currently calls into `preprocess` which handles either path. The fused kernel from c25 will accept GpuMat directly.
- INI: add `UseDirectGpuCapture = false` to all three configs in `dep/`.

#### Verification (mandatory before merge)

- With `UseDirectGpuCapture = false`: behavior identical to c25. The flag is the rollback path.
- With `UseDirectGpuCapture = true`: same detection coords as the `=false` path within ±1 px. FPS measurably higher (target: capture latency drops by 1–3 ms).
- Toggle the flag mid-session if you can — verify clean teardown and re-init.
- Resolution-change / lock-screen / UAC: the c09 access-loss recovery has to work for the CUDA-mapped resource too. Specifically, `cudaGraphicsUnregisterResource` must run before the new `IDXGIOutputDuplication` is created.
- Multi-monitor: the current code only enumerates output 0. If the user multi-monitors, the wrong monitor gets captured in either path — orthogonal issue, document it.

#### Risk register

- **Resource lifecycle**: `cudaGraphicsRegisterResource` must outlive every `cudaGraphicsMapResources` call. On device loss, both must be torn down in order. Triple-check the destructor + recovery path.
- **Format compatibility**: `DXGI_FORMAT_B8G8R8A8_UNORM` is supported by the CUDA-D3D11 interop API; the resulting `cudaArray_t` is read via `cudaArrayLayered` semantics. No surprises on this format specifically.
- **Synchronization**: D3D11 work queue and CUDA stream are separate. Either insert `cudaWaitExternalSemaphoresAsync` (TRT 10 / CUDA 12 has the API) or accept that the unmap/release boundary is the sync point. Implicit sync via map/unmap is what most reference implementations use; that's fine for this project's per-frame cadence.
- **Read-only flag**: register the resource with `cudaGraphicsRegisterFlagsReadOnly` so CUDA never tries to write back. Faster too.

---

### c28 — overlap crosshair GPU match with TRT inference

#### Why

c27 (already landed) moved CS2 crosshair tracking from CPU `cv::matchTemplate` to `cv::cuda::TemplateMatching`. The match runs **synchronously on the default CUDA stream** before `YoloV8::detectObjects()` is called, so it still serializes against TRT inference even though the data dependency is independent — the matcher reads the small ROI and writes a result map; inference reads the full ROI and writes detection logits; they share no inputs and no outputs. They could run concurrently and only sync where the consumer needs the result (`MouseController::aim`, which reads `crosshairPos`).

Estimated win: the entire ~0.2–0.3 ms cost of the GPU match disappears into the inference shadow. Smaller absolute number than c25/c26 since c27 already cut the CPU cost; this is the remainder.

#### Design

Use a separate `cv::cuda::Stream` for the matcher so it runs concurrently with the engine's `m_stream`. Sync only at the consumer site — just before `mouseController->aim(detections)`.

```cpp
// new member of ObjectDetectionSystem
cv::cuda::Stream crosshairStream;

// in detectionThread, replace the synchronous match block:
gpuCrosshairRoi.upload(smallCroppedFrame, crosshairStream);
crosshairMatcher->match(gpuCrosshairRoi, gpuCrosshairTemplate, gpuCrosshairResult, crosshairStream);
// (cv::cuda::minMaxLoc has a stream overload too)
double maxVal = 0.0;
cv::Point maxLoc;
cv::cuda::minMaxLoc(gpuCrosshairResult, nullptr, &maxVal, nullptr, &maxLoc);  // need to wait first

// ... yoloV8->detectObjects(croppedFrame) runs on engine's m_stream concurrently ...

crosshairStream.waitForCompletion();  // sync here, just before consumer
mouseController->setCrosshairPosition(crosshairPos.x, crosshairPos.y);
mouseController->aim(detections);
```

`cv::cuda::minMaxLoc` does not have a documented stream overload in OpenCV's CUDA module — the practical pattern is to have it sync the stream itself (default behavior when called without a stream) and accept that the sync point lands here rather than at the consumer. To get true overlap, use the lower-level `cv::cuda::minMaxLocAsync` (if available in your OpenCV build) or replicate the reduction with a small custom kernel on the same stream.

If c26 has already landed, you can skip the upload entirely — the small ROI becomes a sub-rect view of the GPU frame already mapped from DXGI:

```cpp
// post-c26: frame is already a cv::cuda::GpuMat
cv::cuda::GpuMat gpuSmallRoi = gpuFrame(smallRoi);  // O(1) view, no copy
crosshairMatcher->match(gpuSmallRoi, gpuCrosshairTemplate, gpuCrosshairResult, crosshairStream);
```

That's another ~0.05–0.1 ms saved (the upload).

#### File-level plan

- `src/object_detection_image_mt.cpp`:
  - Add `cv::cuda::Stream crosshairStream` member to `ObjectDetectionSystem`.
  - In `detectionThread()`, pass `crosshairStream` to `gpuCrosshairRoi.upload(...)` and `crosshairMatcher->match(...)`.
  - Move the synchronization (either via `crosshairStream.waitForCompletion()` or via the implicit sync in `cv::cuda::minMaxLoc`) to immediately before the `mouseController->setCrosshairPosition` / `mouseController->aim` calls.
  - Verify `gpuCrosshairResult` lives long enough that the async match completes before the next iteration overwrites it (since `cv::cuda::GpuMat` doesn't refcount per stream, the safe pattern is one result buffer per outstanding match, or sync at the end of the loop).

#### Verification

- Detection latency on CS2 drops by ~0.2–0.3 ms vs c27.
- Crosshair coords identical to c27 within ±1 px on a fixed scene (same matcher, same data, just a different stream).
- No regression during recoil patterns — the sync point is before `mouseController->aim()`, so the aim path always sees a fresh result.

#### Risk register

- **Stream lifetime + result buffer reuse**: if you don't sync per iteration the next iteration's `match()` write races with the current iteration's `minMaxLoc` read. Easy bug — pin the loop to one outstanding match by syncing at the end if you prefer simplicity over maximum overlap.
- **Diminishing return**: this commit is small (~0.2–0.3 ms). If you've also landed c25 and c26, consider whether the overlap is worth the extra stream complexity vs. just leaving c27 synchronous. For this project, latency is the goal so the answer is yes — but it's the smallest win in the deferred set.
- **OpenCV `minMaxLoc` stream support**: verify the `cv::cuda::minMaxLoc` overload your OpenCV build exposes. Older builds had a `cv::cuda::Stream` parameter; newer builds may have moved it to a separate `Async` variant. If neither overlaps cleanly, write a 20-line reduction kernel that reads `gpuCrosshairResult` on `crosshairStream` and writes the peak's `(x, y)` to a 2-int device buffer, then `cudaMemcpyAsync` to host.

---

## 2. INT8 inference path

`Precision::INT8` is declared in `src/EngineBase.h:38`; `src/EngineFactory.h:18` throws "INT8 engine not implemented yet". Adding it gives a 1.4–2× latency win on RTX 40 series at 640×640 (~0.7–1.0 ms vs ~1.2–1.7 ms for FP16). At 320×320 the kernels are memory-bound; INT8 win shrinks to <0.2 ms. So the practical case is **CS2 (640×640) gets a real win, Valorant (320×320) barely benefits**.

### Calibrator approach

NVIDIA deprecated the implicit-calibration path (`IInt8EntropyCalibrator2` etc.) in TensorRT 10.1+ in favor of explicit Q/DQ in the ONNX via NVIDIA's TensorRT Model Optimizer (formerly AMMO). The implicit path still works and is the simplest first pass for a single-network deployment. **Start with `IInt8EntropyCalibrator2`** (entropy minimization, per-channel weight quantization, default for vision/CNN). Switch to explicit Q/DQ later only if accuracy requires QAT.

### Calibration dataset

500–1000 images **drawn from the same distribution as inference** — actual CS2/Valorant gameplay screenshots at the configured capture region and resolution. **Not COCO, not generic in-game art.** Pre-process each through the exact same pipeline as inference (the new fused kernel in c25, or the existing `blobFromGpuMats`). Calibration batch size 8–32; small batches give poor activation histograms.

Where the data lives is up to the user — `Options::calibrationDataDirectoryPath` already exists in the struct (`src/EngineBase.h:72`). Suggested: `dep/calibration_cs2/` and `dep/calibration_valo/` as directories of `.png` or `.jpg`, gitignored, the user populates manually from a recording session.

### Mixed-precision strategy (FP16 + INT8 in the same engine)

`BuilderFlag::kFP16`, `kINT8`, and `kTF32` are independent and can all be set together. With `kINT8 | kFP16`, TRT picks the fastest tactic per layer with FP32 → FP16 → INT8 fallback.

**Keep the detection head FP16.** YOLOv8's final conv layers (the `4 + numClasses` output projection per detect branch) are quantization-sensitive; their xywh regression output is more sensitive than the classification scores. Walk the network after parsing and call `ILayer::setPrecision(DataType::kHALF)` + `ILayer::setOutputType(DataType::kHALF)` on the last conv of each detect branch.

The SiLU-heavy backbone and neck quantize cleanly with PTQ.

### Implementation sketch

`EngineINT8` is mostly a clone of `EngineFP16` because the **engine's external I/O tensors stay FP16** (only weights and intermediate activations are INT8). So `runInference` and the input-prep path are unchanged. The deltas:

```cpp
class EngineINT8 : public EngineBase {
public:
    explicit EngineINT8(const Options &options) : EngineBase(options) {
        if (options.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("EngineINT8 requires Options::calibrationDataDirectoryPath");
        }
    }

protected:
    const char *precisionSuffix() const override { return "int8"; }

    void applyPrecisionFlags(nvinfer1::IBuilder &builder, nvinfer1::IBuilderConfig &config) override {
        if (!builder.platformHasFastInt8()) {
            throw std::runtime_error("GPU does not support fast INT8");
        }
        config.setFlag(nvinfer1::BuilderFlag::kFP16);   // mixed precision: keep some layers FP16
        config.setFlag(nvinfer1::BuilderFlag::kINT8);
        config.setInt8Calibrator(m_calibrator.get());
    }

    // After parsing the network, mark detect-head output convs as FP16.
    // Hook this in EngineBase::buildSerialized between parse and addOptimizationProfile,
    // via a new virtual `applyLayerPrecisionOverrides(network)` defaulting to no-op.

private:
    std::unique_ptr<Int8EntropyCalibrator2> m_calibrator;  // owns the calibration dataset reader
};
```

Plus a `Int8EntropyCalibrator2` class implementing the four virtuals:
- `getBatchSize()` → `Options::calibrationBatchSize`.
- `getBatch(void* bindings[], const char* names[], int nbBindings)` → load N images from the directory, run the same preprocessing as inference (call the c25 fused kernel), copy onto the device pointer in `bindings[0]`. Return false when the dataset is exhausted.
- `readCalibrationCache(size_t&)` / `writeCalibrationCache(const void*, size_t)` → persist the cache to disk so engine rebuilds skip recalibration. Cache filename should include the calibration directory's content hash so swapping the dataset invalidates the cache (mirror the c18 ONNX hash pattern).

**Effort estimate:** 150–250 lines of C++ + a calibration script that captures and saves N gameplay frames.

### Engine cache key

The c18 ONNX hash already in `serializeEngineOptions()` covers the model. For INT8 also include:
- `Options::calibrationBatchSize`
- A hash of the calibration directory contents (FNV-1a over the sorted list of filename + size + mtime would be cheaper than hashing every file's content)

so swapping calibration data invalidates the cached engine.

### Known YOLOv8 INT8 pitfalls

- **SiLU's unbounded positive tail** clips under per-tensor symmetric INT8 and can collapse mAP. Per-channel weight quantization (TRT default) plus a representative calibration set mitigates this. Some teams retrain with ReLU swapped for SiLU before INT8 deployment for ~2× speedup at no accuracy loss — **not recommended for this project** since you'd need to retrain on your CS2/Valorant dataset.
- **Detection head regression accuracy** is the most quantization-sensitive part — keep it FP16 (above).
- **PTQ vs QAT**: start with PTQ. If mAP drops more than 2 points on the user's own validation set, do QAT in PyTorch (Ultralytics supports `torch.ao.quantization`; NVIDIA's `pytorch-quantization` toolkit also works) and re-export with embedded Q/DQ nodes.

### Realistic expectations

- **Latency**: ~0.4–0.7 ms saved per inference at 640×640 on RTX 40, vs ~0.1–0.2 ms at 320×320.
- **mAP**: expect 0.5–1.5 point drop with good calibration data; >2 points means do QAT.
- **End-to-end impact**: if your bottleneck is capture or HID write, the INT8 win is invisible. Profile first.

---

## 3. Newer YOLO versions

YOLOv8 was released early 2023. Several newer versions exist; the question is which is worth migrating to for this real-time gaming detector that prioritizes speed over absolute mAP.

### Landscape (as of 2026)

| Model        | Released    | License | Maintainer        | TRT 10 status |
|--------------|-------------|---------|-------------------|---------------|
| YOLOv9       | Feb 2024    | GPL-3   | Wang et al. + community ports | Mature |
| YOLOv10      | May 2024    | AGPL-3  | Tsinghua THU-MIG, distributed via Ultralytics | Mature |
| YOLO11       | Sep 2024    | AGPL-3  | Ultralytics       | First-class |
| YOLO12       | Feb 2025    | AGPL-3  | Ultralytics       | First-class, marketed as research-oriented |
| YOLO26       | Jan 2026    | AGPL-3  | Ultralytics       | First-class, two open export bugs (see below) |

### Nano-tier comparison (640 input, COCO val)

| Model    | Params | FLOPs | mAP-val | T4 TRT10 FP16 (official) |
|----------|--------|-------|---------|---------------------------|
| YOLOv8n  | 3.2M   | 8.7G  | 37.3    | ~1.5 ms                   |
| YOLOv9t  | 2.0M   | 7.7G  | 38.3    | ~2.0 ms                   |
| YOLOv10n | 2.3M   | 6.7G  | 39.5    | 1.84 ms                   |
| **YOLO11n** | 2.6M | 6.5G  | 39.5    | **1.5 ms**                |
| YOLO12n  | 2.6M   | 6.5G  | 40.6    | 1.64 ms                   |
| YOLO26n (one-to-many) | 2.4M | 5.4G | 40.9 | 1.7 ms     |
| YOLO26n (end-to-end)  | 2.4M | 5.4G | 40.1 | 1.7 ms     |

T4 TRT10 numbers come from the official Ultralytics docs pages. RTX 40 numbers scale roughly proportionally — YOLO11n's GPU-FP16 lead over YOLO26n persists. The much-publicized "YOLO26 is 43% faster" headline is **CPU latency**, not GPU — irrelevant for this RTX 40 pipeline.

YOLO11n is the Pareto winner in the nano tier on GPU FP16: lowest latency, identical mAP to v10n, smoothest ONNX/TRT 10 path.

### Output-shape compatibility with the existing C++ postprocess

`src/yolov8.cpp::postprocessDetect` (post-c24) walks the layout `[1, 4 + numClasses, numAnchors]` — channel-major, anchors as the last dim. Specifically:

- **YOLO11n / YOLO12n / YOLOv9** → identical layout. **`yolov8.cpp` is drop-in.** No C++ changes.
- **YOLOv10** → defaults to NMS-free output shape `[1, 300, 6]` (top-300 detections, each `xyxy + score + class`). The existing postprocess + NMS path **breaks**. You'd either:
  - (a) export with `nms=False` to get the v8-style raw output, losing the latency win that's the point of v10.
  - (b) write a new ~30-line postprocess: iterate the 300 rows, threshold on score, no NMS at all.
- **YOLO26** → exports either layout, controlled by the `end2end` flag:
  - `end2end=True` (default): `[1, 300, 6]` xyxy + score + class — like YOLOv10. Breaks `postprocessDetect`.
  - `end2end=False`: `[1, 4 + nc, 8400]` — same layout as v8/v11. **`yolov8.cpp` is drop-in.**
  - The flags `end2end` and `nms` are mutually exclusive; passing `nms=True` forces `end2end=False`.

### Export / training pipeline impact

`scripts/pytorch2onnx.py` currently uses `opset=12, simplify=True`. For v10/v11/v12/v26:

- **Bump opset to 17.** Ultralytics defaults to 17 in 2025+, and YOLO12's attention ops require ≥17. No further bumps documented for v26.
- `simplify=True` still requires `onnxsim` installed — keep it.
- `dynamic=True` is still incompatible with `half=True`; the existing FP16 export path which forces `dynamic=False` is unchanged.
- `from ultralytics import YOLO; model.export(format="onnx", ...)` API is unchanged across v8/v9/v10/v11/v12/v26 — same script structure.
- Training data and INI labels are reusable verbatim — datasets transfer without remapping.

### YOLO26-specific export gotchas (early 2026)

YOLO26 has two open Ultralytics issues as of early 2026 that affect ONNX/TRT export:

- **#23645 — `end2end=True, half=True` produces FP32 output0** instead of honoring `half=True`. Workaround: use `end2end=False` (which you'd want anyway for drop-in C++ compat). PR #23759 references a fix; verify your installed version actually contains it.
- **#23756 — `aten::index` opset-20 warning** on TRT export with `end2end=True`: "If indices include negative values, the exported graph will produce incorrect results." Engine still builds and runs, but the warning is unresolved. Does not appear with `end2end=False`.

Both bugs disappear in the `end2end=False` path. No EfficientNMS plugin required for that path (NMS happens in C++ as today).

### INT8/QAT note for YOLO26

YOLO26's architectural changes (DFL removal, NMS-free decoder option) are **friendly to quantization** — fewer custom ops to wrap, less precision-sensitive logic in the head. Ultralytics markets v26 as "QAT-ready" but there is **no documented native `qat=True` flag** in the export pipeline as of early 2026; INT8 is still post-training-only via TensorRT calibration (the same path the [INT8 section above](#2-int8-inference-path) describes). On TensorRT ≤ 10.3.0, `int8=True` auto-disables `end2end` — another reason to default to `end2end=False`.

### Recommendation

**Switch to YOLO11n + FP16, retrained on the user's CS2/Valorant datasets.** Reasoning:

- **Lowest GPU FP16 latency** in the nano tier (1.5 ms T4 TRT10 vs 1.7 ms for YOLO26n). For a latency-critical real-time loop this is the load-bearing metric.
- Output layout is byte-identical to YOLOv8 → `yolov8.cpp`, `EngineFP16`, and the build pipeline keep working with **zero C++ changes**.
- ~+2 mAP over YOLOv8n. The +1.4 mAP further bump from YOLO26n is unlikely to matter for a tiny 2-class (Valorant) or 4-class (CS2) detector that's already well-saturated on a fine-tuned dataset.
- Same Ultralytics API, same training pipeline, same AGPL-3 status as v8 — no license disruption.
- First-class TensorRT 10 support, no open export bugs.

**Skip YOLOv10** unless you specifically want NMS-free inference — it forces a postprocess rewrite for marginal gain over YOLO11n.

**Skip YOLO12** unless you need the last mAP point — Ultralytics themselves position it as research-oriented over production-oriented.

**Skip YOLO26** for now. It's slightly slower at GPU FP16 in the nano tier, the headline wins (CPU latency, edge/INT8 deployment) don't apply to this pipeline, and the export pipeline still has open bugs ~4 months post-release. **Reconsider when:** (a) you want to add a real INT8 path and the DFL removal makes calibration cleaner — verify by testing both v11n-INT8 and v26n-INT8 mAP on your own dataset, or (b) Ultralytics ships a tagged release with #23645 and #23756 closed and a clean changelog.

If you do INT8 ([above](#2-int8-inference-path)) and YOLO11n in the same migration, do them in this order: (1) switch to YOLO11n FP16, validate detection quality with new training, (2) add INT8 path, (3) calibrate INT8 against the YOLO11n model. Don't combine both into a single change — too many variables when something regresses.

### File-level plan for the YOLO11 migration

- `scripts/pytorch2onnx.py:73-78` — add `v11n`/`v11s`/`v11m`/`v11l`/`v11x` to the choices, bump default opset to 17.
- Train YOLO11n on your existing CS2 and Valorant datasets (not in this repo's scope; happens in your training environment).
- Place the new `.onnx` files in `dep/`, e.g. `yolo11n_val_fp16.onnx`, `yolo11n_cs2.onnx`.
- Update `dep/config_*.ini` `ModelPath` entries to point at the new ONNX files. `Labels`, `HeadLabelID*`, and everything else stays the same.
- The c18 ONNX-content hash will automatically rebuild a fresh TensorRT engine plan. No engine-file management needed.

That's the entire migration if everything goes well.

### File-level plan if you ignore the recommendation and try YOLO26

If you want to test YOLO26n anyway:

- Same `pytorch2onnx.py` and `dep/config_*.ini` edits as the YOLO11 plan.
- **Force `end2end=False`** in the export call. The flag passes straight through `model.export(...)`. This sidesteps both open export bugs and keeps `yolov8.cpp` drop-in.
- Do not pass `nms=True` (it's mutually exclusive with `end2end`); the project's existing NMS in `postprocessDetect` is the NMS path.
- If you also want INT8 with YOLO26: don't pass `int8=True` to Ultralytics' export — let TRT calibration in `EngineINT8` handle it (the path described [above](#2-int8-inference-path)). On TRT ≤ 10.3.0, `int8=True` would force `end2end` off anyway.
- Track Ultralytics issues #23645 and #23756 — if both close before you do this, the constraints relax.

---

## What this document is not

- A schedule. Land these in whatever order makes sense, after the cleanup PR is merged.
- A guarantee on perf numbers. The latencies above are public benchmark estimates; measure on the user's actual hardware before celebrating.
- An exhaustive perf hunt. Other small wins (e.g., reusing the FP16 staging buffer across both input and output, prefetching the next frame's GpuMat, profiling the NMS step) live in code-review territory, not in a follow-up doc.
