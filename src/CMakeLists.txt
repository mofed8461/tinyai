# CMake entry point
cmake_minimum_required (VERSION 3.5)
project (TinyAI C CXX)

macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)

use_cxx11()

add_library(TinyAI STATIC
	tinyann.h
	tinyann.cpp
	tinyneat.h
	tinyneat.cpp)

target_link_libraries(TinyAI) 

