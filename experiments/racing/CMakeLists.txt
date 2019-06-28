# CMake entry point
cmake_minimum_required (VERSION 3.5)
project (TinyAI_racing)



add_executable(Racing
	level.h
	level.cpp
	main.cpp
	physics.h
	TINY/tinystr.h
	TINY/tinystr.cpp
	TINY/tinyxml.h
	TINY/tinyxml.cpp
	TINY/tinyxml2.h
	TINY/tinyxml2.cpp
	TINY/tinyxmlerror.cpp
	TINY/tinyxmlparser.cpp
   )

target_link_libraries(Racing ${ALL_LIBS})
