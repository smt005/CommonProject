cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE INCLUDES ${CUDA_DIR}/Source/*.h)

if (CUDA_FOUND)
	file(GLOB_RECURSE SOURCES ${CUDA_DIR}/Source/*.cu)
else()
	file(GLOB_RECURSE SOURCES ${CUDA_DIR}/Source/*.cpp)
Endif()

source_group(TREE ${CUDA_DIR} TREE PREFIX "" FILES ${INCLUDES} ${SOURCES})

# ThirdPathy
include_directories(${GAME_SOURCE})
include_directories(${ENGINE_SOURCE})
include_directories(${GLM_INCLUDES})
include_directories(${JSONCPP_INCLUDES})

if (CUDA_FOUND)
	CUDA_ADD_LIBRARY(${CUDA_NAME} STATIC ${INCLUDES} ${SOURCES})
	set_target_properties(${CUDA_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
	add_library(${CUDA_NAME} STATIC ${INCLUDES} ${SOURCES})
Endif()
