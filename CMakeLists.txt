cmake_minimum_required(VERSION 3.22)
project(YOLOv8_TensorRT_CPP LANGUAGES CXX CUDA)

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

# Set the runtime library for MSVC
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif()

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set TensorRT paths using CUDA installation
set(TENSORRT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" CACHE PATH "Path to TensorRT installation")
set(TENSORRT_INCLUDE_DIRS "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")

# Find TensorRT libraries with version numbers
file(GLOB NVINFER_LIB "${TENSORRT_LIB_DIR}/nvinfer_10.lib")
file(GLOB NVONNXPARSER_LIB "${TENSORRT_LIB_DIR}/nvonnxparser_10.lib")
file(GLOB NVINFER_PLUGIN_LIB "${TENSORRT_LIB_DIR}/nvinfer_plugin_10.lib")

if(NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB OR NOT NVINFER_PLUGIN_LIB)
    message(FATAL_ERROR "Could not find TensorRT libraries. Please check your installation.")
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

# Find DirectX libraries
if(WIN32)
    find_path(D3D_COMPILER_DIR NAMES d3dcompiler.lib PATHS "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/um/x64")
    if(D3D_COMPILER_DIR)
        set(D3D11_LIB ${D3D_COMPILER_DIR}/d3d11.lib)
        set(D3DCOMPILER_LIB ${D3D_COMPILER_DIR}/d3dcompiler.lib)
        set(DXGI_LIB ${D3D_COMPILER_DIR}/dxgi.lib)
    else()
        message(FATAL_ERROR "Could not find DirectX libraries. Make sure the DirectX SDK is installed.")
    endif()
endif()

# Set CUDA specific flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_89")  # Adjust SM architecture as needed

# Create CUDA kernels library
add_library(engine_cuda_kernels STATIC
    src/EngineFP16Kernels.cu
)

set_target_properties(engine_cuda_kernels PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(engine_cuda_kernels PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
)

# Create software fuser library
add_library(fusion_cuda STATIC
    src/SoftwareFuser.cu
)

set_target_properties(fusion_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create main YOLOv8 library
add_library(yolov8_tensorrt STATIC
    src/yolov8.cpp
)

# Create main executable
add_executable(detect_object_image
    src/object_detection_image_mt.cpp
    src/MouseController.cpp
)

# Set compile definitions
target_compile_definitions(detect_object_image PRIVATE
    CUDA_HAS_FP16=1
    ENABLE_DEBUG=1
    USE_CUDA=1
)

# Link CUDA runtime to libraries
target_link_libraries(engine_cuda_kernels PUBLIC
    CUDA::cudart
)

target_link_libraries(fusion_cuda PUBLIC
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
    fusion_cuda
    ${OpenCV_LIBS} 
    CUDA::cudart
    CUDA::cuda_driver
    # TensorRT libraries
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${NVINFER_PLUGIN_LIB}
    # Windows-specific libraries
    setupapi 
    hid 
    d3d11 
    dxgi 
    Dwmapi.lib
    ${D3D11_LIB}
    ${D3DCOMPILER_LIB}
    ${DXGI_LIB}
)

# Set output directories
set_target_properties(detect_object_image PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
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
if(WIN32)
    file(GLOB TENSORRT_DLLS "${TENSORRT_LIB_DIR}/*.dll")
    add_custom_command(TARGET detect_object_image POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TENSORRT_DLLS}
            $<TARGET_FILE_DIR:detect_object_image>
    )
endif()

if(WIN32)
    file(GLOB DEP_FILES "${CMAKE_SOURCE_DIR}/dep/*.*")
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

# Optional: Enable testing
enable_testing()