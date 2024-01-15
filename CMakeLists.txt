cmake_minimum_required(VERSION 3.27)
project(nn)

set(CMAKE_CXX_STANDARD 20) # Changed to c++2a to match the Makefile's standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++ -pthread") # Added basic libraries

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add native processor optimizations for release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# The following code checks the version of gcc and links the filesystem library if the version is less than 9
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++fs")
  endif()
endif()

file(GLOB_RECURSE nn_SRC
        "src/*.h"
        "src/*.cpp"
)
add_executable(nn ${nn_SRC})

find_package(fmt CONFIG REQUIRED)
target_link_libraries(nn PRIVATE fmt::fmt-header-only)
