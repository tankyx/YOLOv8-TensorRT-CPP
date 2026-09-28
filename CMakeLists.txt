cmake_minimum_required(VERSION 3.22)
project(YOLOv8_TensorRT_CPP LANGUAGES C CXX CUDA)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Enable Hot Reload for MSVC compilers if supported
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Set the runtime library for MSVC
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif()

# Find required packages
if(WIN32)
    find_package(OpenCV REQUIRED)
else()
    # Only the modules actually used — pulling in the full set drags libopencv_viz,
    # whose VTK dependency may be missing.
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui dnn
                                            cudaarithm cudaimgproc cudawarping cudafilters)
endif()
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(WIN32)
    # Set TensorRT paths using CUDA installation
    set(TENSORRT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" CACHE PATH "Path to TensorRT installation")
    set(TENSORRT_INCLUDE_DIRS "${TENSORRT_ROOT}/include")
    set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")
    set(TENSORRT_BIN_DIR "${TENSORRT_ROOT}/bin")

    # Find TensorRT libraries with version numbers
    file(GLOB NVINFER_LIB "${TENSORRT_LIB_DIR}/nvinfer_10.lib")
    file(GLOB NVONNXPARSER_LIB "${TENSORRT_LIB_DIR}/nvonnxparser_10.lib")
    file(GLOB NVINFER_PLUGIN_LIB "${TENSORRT_LIB_DIR}/nvinfer_plugin_10.lib")

    if(NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB OR NOT NVINFER_PLUGIN_LIB)
        message(FATAL_ERROR "Could not find TensorRT libraries. Please check your installation.")
    endif()
else()
    # Linux: TensorRT from system packages (AUR `tensorrt`) or TensorRT_DIR hint
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(TensorRT REQUIRED)
    find_library(NVINFER_PLUGIN_LIB NAMES nvinfer_plugin)
    if(NOT NVINFER_PLUGIN_LIB)
        message(FATAL_ERROR "Could not find nvinfer_plugin. Please check your TensorRT installation.")
    endif()
    set(TENSORRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIRS})
    set(NVINFER_LIB ${TensorRT_LIBRARY})
    set(NVONNXPARSER_LIB ${TensorRT_NVONNXPARSER_LIBRARY})

    # Screen capture (PipeWire + xdg-desktop-portal) dependencies
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PIPEWIRE REQUIRED IMPORTED_TARGET libpipewire-0.3)
    pkg_check_modules(DBUS REQUIRED IMPORTED_TARGET dbus-1)

    # Debug overlay (wlr-layer-shell on Wayland)
    pkg_check_modules(WAYLAND REQUIRED IMPORTED_TARGET wayland-client)
endif()

message(STATUS "Found TensorRT libraries:")
message(STATUS "  nvinfer: ${NVINFER_LIB}")
message(STATUS "  nvonnxparser: ${NVONNXPARSER_LIB}")
message(STATUS "  nvinfer_plugin: ${NVINFER_PLUGIN_LIB}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS} 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
)

# Set CUDA architecture (default sm_89 / RTX 40 series on Windows;
# 'native' detection on Linux — override with -DCMAKE_CUDA_ARCHITECTURES=XX)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(WIN32)
        set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for")
    else()
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
    endif()
endif()

# Create CUDA kernels library
add_library(engine_cuda_kernels STATIC
    src/EngineFP16Kernels.cu
    src/YoloPostprocKernels.cu
)

set_target_properties(engine_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(engine_cuda_kernels PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/bigobj>
)

# Create software fuser library (Windows-only: depends on DXGI/D3D11 capture
# and DirectShow capture-card input; unused by the main executable on Linux)
if(WIN32)
    add_library(fusion_cuda STATIC
        src/SoftwareFuser.cu
    )

    set_target_properties(fusion_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(fusion_cuda PUBLIC
        CUDA::cudart
    )
endif()

# Create main YOLOv8 library
add_library(yolov8_tensorrt STATIC
    src/YoloDetector.cpp
    src/yolov8.cpp
    src/YoloV11.cpp
    src/YoloV26.cpp
    src/DetectorFactory.cpp
)

# Create main executable
add_executable(detect_object_image
    src/object_detection_image_mt.cpp
    src/MouseController.cpp
)
if(NOT WIN32)
    target_sources(detect_object_image PRIVATE
        src/LinuxCapture.cpp
        src/LinuxOverlay.cpp
        src/wayland/xdg-shell-protocol.c
        src/wayland/wlr-layer-shell-unstable-v1-protocol.c
    )
endif()

# Link CUDA runtime to libraries
target_link_libraries(engine_cuda_kernels PUBLIC
    CUDA::cudart
)

target_link_libraries(yolov8_tensorrt PUBLIC
    ${OpenCV_LIBS}
    CUDA::cudart
    CUDA::cuda_driver
    engine_cuda_kernels
)

# Link everything to the main executable
target_link_libraries(detect_object_image PRIVATE
    yolov8_tensorrt
    ${OpenCV_LIBS}
    CUDA::cudart
    CUDA::cuda_driver
    # TensorRT libraries
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${NVINFER_PLUGIN_LIB}
)

if(WIN32)
    target_link_libraries(detect_object_image PRIVATE fusion_cuda)
endif()

if(WIN32)
    # Windows-specific libraries (DXGI + D3D11 used by DXGICapture; HID/setupapi by MouseController)
    target_link_libraries(detect_object_image PRIVATE setupapi hid d3d11 dxgi)
else()
    target_link_libraries(detect_object_image PRIVATE PkgConfig::PIPEWIRE PkgConfig::DBUS PkgConfig::WAYLAND)
endif()

# Set output directories
set_target_properties(detect_object_image PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set_target_properties(yolov8_tensorrt PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Install rules
install(TARGETS detect_object_image
    RUNTIME DESTINATION bin
)

install(TARGETS yolov8_tensorrt
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Copy required DLLs to output directory
# TensorRT DLLs may live alongside .lib (original NVIDIA zip layout) or in
# bin/ (when merged into the CUDA toolkit tree).
if(WIN32)
    file(GLOB TENSORRT_DLLS
        "${TENSORRT_LIB_DIR}/nvinfer*.dll"
        "${TENSORRT_LIB_DIR}/nvonnx*.dll"
        "${TENSORRT_BIN_DIR}/nvinfer*.dll"
        "${TENSORRT_BIN_DIR}/nvonnx*.dll"
    )
    if(TENSORRT_DLLS)
        add_custom_command(TARGET detect_object_image POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TENSORRT_DLLS}
                $<TARGET_FILE_DIR:detect_object_image>
        )
    endif()
endif()

# Copy INI configs / ONNX models / assets next to the executable so relative
# paths in the configs resolve from the bin directory.
file(GLOB DEP_FILES "${CMAKE_SOURCE_DIR}/dep/*.*")
if(DEP_FILES)
    add_custom_command(TARGET detect_object_image POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DEP_FILES}
            $<TARGET_FILE_DIR:detect_object_image>
    )
endif()

# Optional: Print the files that will be copied
foreach(DEP_FILE ${DEP_FILES})
    message(STATUS "Will copy dependency: ${DEP_FILE}")
endforeach()