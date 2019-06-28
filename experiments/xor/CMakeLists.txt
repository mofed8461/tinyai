# CMake entry point
cmake_minimum_required (VERSION 3.5)
project (TinyAI_xor)

include_directories(../../src/)

set(ALL_LIBS
	TinyAI
)


add_executable(XOR
	xor.cpp
)

target_link_libraries(XOR ${ALL_LIBS})

