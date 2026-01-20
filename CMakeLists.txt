cmake_minimum_required(VERSION 3.27)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 指定生成类型
set(CMAKE_BUILD_TYPE "Debug")

# # 设置编译器
# set(CMAKE_C_COMPILER D:/MinGW-Posix/bin/gcc.exe)
# set(CMAKE_CXX_COMPILER D:/MinGW-Posix/bin/g++.exe)

# 设置CXX标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(CmakeCpp)

# 引入自定义函数
include(${PROJECT_SOURCE_DIR}/func.cmake)
message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Project Dir: ${CMAKE_CURRENT_SOURCE_DIR}")


# 设置链接库安装目录
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)

file(COPY ${PROJECT_SOURCE_DIR}/README.md DESTINATION ${PROJECT_SOURCE_DIR}/install)

# 进入主工程文件夹
add_subdirectory(src)