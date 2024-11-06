cmake_minimum_required(VERSION 3.22)
project(YOLOv8_TensorRT_CPP LANGUAGES CXX CUDA)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

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

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} 
                   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                   ${CUDA_INCLUDE_DIRS})

# Add custom target to build shaders
add_custom_target(Shaders ALL DEPENDS ${SHADERS})

# Find DirectX libraries
find_path(D3D_COMPILER_DIR NAMES d3dcompiler.lib PATHS "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/um/x64")

# Set the library paths
if(D3D_COMPILER_DIR)
    set(D3D11_LIB ${D3D_COMPILER_DIR}/d3d11.lib)
    set(D3DCOMPILER_LIB ${D3D_COMPILER_DIR}/d3dcompiler.lib)
    set(DXGI_LIB ${D3D_COMPILER_DIR}/dxgi.lib)
else()
    message(FATAL_ERROR "Could not find DirectX libraries. Make sure the DirectX SDK is installed.")
endif()

# Set CUDA specific flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_89")  # Adjust SM architecture as needed

# Create CUDA library for the fusion code
add_library(fusion_cuda STATIC
    src/SoftwareFuser.cu
)

set_target_properties(fusion_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(fusion_cuda
    CUDA::cudart
)

# Add main executable
add_executable(detect_object_image 
    src/object_detection_image_mt.cpp 
    src/yolov8.cpp 
    src/engine.cpp 
    src/MouseController.cpp
)

add_dependencies(detect_object_image Shaders)

target_link_libraries(detect_object_image 
    fusion_cuda  # Add the CUDA library
    ${OpenCV_LIBS} 
    CUDA::cudart
    CUDA::cuda_driver
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/nvinfer_10.lib"
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/nvonnxparser_10.lib"
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/nvinfer_plugin_10.lib"
    setupapi 
    hid 
    d3d11 
    dxgi 
    Dwmapi.lib
    ${D3D11_LIB}
    ${D3DCOMPILER_LIB}
    ${DXGI_LIB}
)

# Set the output directory for the executable
set_target_properties(detect_object_image PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)