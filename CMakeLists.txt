cmake_minimum_required(VERSION 3.22)
project(YOLOv8_TensorRT_CPP LANGUAGES CXX)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the runtime library for MSVC
if (MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif()

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# Function to compile HLSL shaders
function(compile_shader shader_file target profile entry_point)
    get_filename_component(shader_name ${shader_file} NAME_WE)
    set(output_file ${CMAKE_BINARY_DIR}/${shader_name}_${entry_point}.cso)
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND fxc.exe /T ${profile} /E ${entry_point} /Fo ${output_file} ${CMAKE_SOURCE_DIR}/${shader_file}
        MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/${shader_file}
        COMMENT "Compiling ${shader_file} (${entry_point})..."
        VERBATIM
    )
    set(${target} ${${target}} ${output_file} PARENT_SCOPE)
endfunction()

# Compile vertex and pixel shaders
set(SHADERS)
compile_shader("src/shader.hlsl" SHADERS "vs_5_0" "VS")
compile_shader("src/shader.hlsl" SHADERS "ps_5_0" "PS")

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

# Add main executable
add_executable(detect_object_image 
    src/object_detection_image_mt.cpp 
    src/yolov8.cpp 
    src/engine.cpp 
    src/MouseController.cpp
    
)

add_dependencies(detect_object_image Shaders)

target_link_libraries(detect_object_image 
    ${OpenCV_LIBS} ${CUDA_LIBRARIES}
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/nvinfer_10.lib"
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/nvonnxparser_10.lib"
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/nvinfer_plugin_10.lib"
    setupapi hid d3d11 dxgi Dwmapi.lib
    ${D3D11_LIB}
    ${D3DCOMPILER_LIB}
    ${DXGI_LIB})