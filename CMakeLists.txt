cmake_minimum_required(VERSION 3.22)
project(YOLOv8_TensorRT_CPP)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMake Options
option(ENABLE_BENCHMARKS "Benchmark the preprocessing, inference, and post processing" OFF)
if (ENABLE_BENCHMARKS)
    add_compile_definitions(ENABLE_BENCHMARKS)
endif()

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# Build and link the executables
add_executable(detect_object_image src/object_detection_image.cpp src/yolov8.cpp src/engine.cpp "src/MouseController.h")
target_link_libraries(detect_object_image PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES}
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/nvinfer_10.lib"
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/nvonnxparser_10.lib"
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/nvinfer_plugin_10.lib")


# Add /NODEFAULTLIB to avoid conflicts
set_target_properties(detect_object_image PROPERTIES LINK_FLAGS "/NODEFAULTLIB:LIBCMT")