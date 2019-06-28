# CMake entry point
cmake_minimum_required (VERSION 3.5)
project (TinyAI)

add_subdirectory(src)


option(TINYAI_BUILD_EXPERIMENTS "TINYAI_BUILD_EXPERIMENTS" ON)

if (TINYAI_BUILD_EXPERIMENTS)
	add_subdirectory(experiments)
endif()