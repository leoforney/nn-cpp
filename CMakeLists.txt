cmake_minimum_required(VERSION 3.22)
project(nn LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -lstdc++ -pthread")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        master
)

FetchContent_MakeAvailable(fmt)

set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++fs")
  endif()
endif()

file(GLOB_RECURSE nn_SRC
        "src/*.h"
        "src/*.cpp"
        "src/*.cu"
)

add_executable(nn ${nn_SRC})

set_target_properties(nn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(nn fmt::fmt-header-only)