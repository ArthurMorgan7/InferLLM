cmake_minimum_required(VERSION 3.15.2)

project(InferLLM LANGUAGES C CXX ASM)

set(CMAKE_CXX_FLAGS " -std=c++11 -pthread -Wno-multichar -fPIC ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS " -g -O0 ${CMAKE_CXX_FLAGS}")
# set(CMAKE_CXX_FLAGS " -O3 -g ${CMAKE_CXX_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")


# 添加预定义宏，管理条件编译代码
add_definitions(-DINFER_PROFILE) 

# 收集源文件
file(GLOB_RECURSE SRC src/*.cpp src/*.h)

# 构建静态库
add_library(InferLLM STATIC ${SRC} ${PLTSRC})

include_directories(
    include/common
    include/core
    include/graph
    include/kern
)

# 使用该库时头文件的搜索路径，之后就不需要给可执行文件添加搜索路径了
target_include_directories(InferLLM PUBLIC include src)


# 生成可执行文件
add_executable(chatglm application/chatglm/chatglm.cpp)
target_link_libraries(chatglm InferLLM)

add_executable(quantizer application/quantizer.cpp)
target_link_libraries(quantizer InferLLM)


# if(ENABLE_TEST)
#   message(STATUS "Build with TEST.")
#   add_subdirectory(third-party/googletest ${CMAKE_CURRENT_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)

#   file(GLOB TEST_SRC test/*.cpp test/CPU/*.cpp test/CPU/*.h)
#   if(ENABLE_GPU)
#     file(GLOB_RECURSE GPU_TEST_SRC test/GPU/*.cpp test/GPU/*.h)
#     list(APPEND TEST_SRC ${GPU_TEST_SRC})
#   endif()
#   add_executable(InferLLMTest ${TEST_SRC})
#   target_include_directories(InferLLMTest PUBLIC test inlude src)
#   target_link_libraries(InferLLMTest InferLLM gtest)
# endif()
