cmake_minimum_required(VERSION 3.30)

project(yolo11)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# Setzt die CUDA-Architektur passend zur GPU
set(CMAKE_CUDA_ARCHITECTURES 86 89)  # auch Mehrere möglich
# Beispiel für NVIDIA Ampere (z. B. RTX 3090)
# https://developer.nvidia.com/cuda-gpus


# Standardpfad für CUDA 12.6 
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message("Setze CUDA Pfade unter Windows")
  set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
else()
  message("Setze CUDA Pfade unter Linux")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")

  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Windows System erkannt")

    # CUDA-Pfade unter Windows
    include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include")
    link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64")

    # TensorRT-Pfade unter Windows
    include_directories("C:/Program Files/NVIDIA/TensorRT-10.9.0.34/include")
    link_directories("C:/Program Files/NVIDIA/TensorRT-10.9.0.34/lib")
    #include_directories("C:/Program Files/NVIDIA/TensorRT-8.6.1.6/include") # TensorRT 8.x
    #link_directories("C:/Program Files/NVIDIA/TensorRT-8.6.1.6/lib") # TensorRT 8.x

  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Linux System erkannt")

    # CUDA-Pfade fuer Windows
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)

    # TensorRT-Pfade fuer Windows
    include_directories(/workspace/shared/TensorRT-8.6.1.6/include) # TensorRT 8.x
    link_directories(/workspace/shared/TensorRT-8.6.1.6/lib) # TensorRT 8.x
  
  else()
    message(WARNING "Unbekanntes Betriebssystem. CUDA/TensorRT-Pfade müssen manuell gesetzt werden!")
  endif()

endif()

add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer_10 cudart) # TensorRT 10.x
#target_link_libraries(myplugins nvinfer cudart) # TensorRT 8.x

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(yolo11_det ${PROJECT_SOURCE_DIR}/yolo11_det.cpp ${SRCS})
target_link_libraries(yolo11_det nvinfer_10)
#target_link_libraries(yolo11_det nvinfer) # TensorRT 8.x
target_link_libraries(yolo11_det cudart)
target_link_libraries(yolo11_det myplugins)
target_link_libraries(yolo11_det ${OpenCV_LIBS})

# Aenderung FP 24.03.2025 - auskommentiert, da nicht benötigt
#add_executable(yolo11_cls ${PROJECT_SOURCE_DIR}/yolo11_cls.cpp ${SRCS})
#target_link_libraries(yolo11_cls nvinfer_10)
#target_link_libraries(yolo11_cls cudart)
#target_link_libraries(yolo11_cls myplugins)
#target_link_libraries(yolo11_cls ${OpenCV_LIBS})

add_executable(yolo11_seg ${PROJECT_SOURCE_DIR}/yolo11_seg.cpp ${SRCS})
target_link_libraries(yolo11_seg nvinfer_10)
#target_link_libraries(yolo11_seg nvinfer) # TensorRT 8.x
target_link_libraries(yolo11_seg cudart)
target_link_libraries(yolo11_seg myplugins)
target_link_libraries(yolo11_seg ${OpenCV_LIBS})

# Aenderung FP 24.03.2025 - auskommentiert, da nicht benötigt
#add_executable(yolo11_pose ${PROJECT_SOURCE_DIR}/yolo11_pose.cpp ${SRCS})
#target_link_libraries(yolo11_pose nvinfer_10)
#target_link_libraries(yolo11_pose cudart)
#target_link_libraries(yolo11_pose myplugins)
#target_link_libraries(yolo11_pose ${OpenCV_LIBS})
