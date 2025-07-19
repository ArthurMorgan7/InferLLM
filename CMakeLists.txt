cmake_minimum_required(VERSION 3.18)

project(InferLLM LANGUAGES C CXX CUDA ASM)

option(ENABLE_GPU "Build with GPU." ON)

set(CMAKE_CXX_FLAGS " -std=c++14 -pthread -Wno-multichar -fPIC ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS " -g -O0 ${CMAKE_CXX_FLAGS}")

set(CMAKE_CUDA_ARCHITECTURES 75)

file(GLOB_RECURSE SRC src/*.cpp src/*.h include/*.hpp)
add_library(InferLLM STATIC ${SRC})

include_directories(
    include/common
    include/core
    include/graph
    include/kernel
)

target_include_directories(InferLLM PUBLIC include src)

if(ENABLE_GPU)
    add_definitions(-DENABLE_GPU=1)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})

    file(GLOB_RECURSE GPU_SRC src/kernel/gpu/*.cu src/kernel/gpu/*.h)
    add_library(InferLLMGPU STATIC ${GPU_SRC})
    target_include_directories(InferLLMGPU PUBLIC include src ${CUDA_CUBLAS_INCLUDE_DIRS})
    target_link_libraries(InferLLMGPU ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    set_target_properties(InferLLMGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(chatglm app/chatglm.cpp)
target_link_libraries(chatglm InferLLM)

if(ENABLE_GPU)
    target_sources(chatglm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/gpu/kernel.cu
    )
    target_link_libraries(InferLLM InferLLMGPU)
    target_link_libraries(chatglm InferLLMGPU)
endif()
