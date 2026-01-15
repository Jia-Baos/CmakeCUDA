cmake_minimum_required(VERSION 3.27)

project(cuda_002)

enable_language(CUDA)

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Project Dir: ${CMAKE_CURRENT_SOURCE_DIR}")

# 查找CUDAToolkit包
find_package(CUDAToolkit REQUIRED)

if (CUDAToolkit_FOUND)
	message(STATUS "CUDAToolkit_FOUND=${CUDAToolkit_FOUND}")
	message(STATUS "CUDAToolkit_VERSION=${CUDAToolkit_VERSION}" )
	message(STATUS "CUDAToolkit_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}")
	message(STATUS "CUDAToolkit_VERSION_MINOR=${CUDAToolkit_VERSION_MINOR}")
	message(STATUS "CUDAToolkit_VERSION_PATCH=${CUDAToolkit_VERSION_PATCH}")
	message(STATUS "CUDAToolkit_BIN_DIR=${CUDAToolkit_BIN_DIR}")
	message(STATUS "CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}")
	message(STATUS "CUDAToolkit_LIBRARY_DIR=${CUDAToolkit_LIBRARY_DIR}")
	message(STATUS "CUDAToolkit_LIBRARY_ROOT=${CUDAToolkit_LIBRARY_ROOT}")
	message(STATUS "CUDAToolkit_TARGET_DIR=${CUDAToolkit_TARGET_DIR}")
	message(STATUS "CUDAToolkit_NVCC_EXECUTABLE=${CUDAToolkit_NVCC_EXECUTABLE}")
else()
	message(STATUS "Could not find CUDAToolkit")
endif()

# 将源代码添加到此项目的可执行文件。
file(GLOB main_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
add_executable(${PROJECT_NAME} ${main_files})

# 链接库头文件路径
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# 链接库文件夹
# 如果动态链接库和静态链接库同名，优先调用静态链接库
target_link_libraries(${PROJECT_NAME} CUDA::cudart)

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)