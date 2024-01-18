# CprE 487 / 587 ML C++ Framework
Written by: Matthew Dwyer, Leo Forney

This repository contains the source code for the C++ ML framework.

## Navigation
TODO

## Prerequisites
Before you can build the framework, ensure that you have the following installed:
- `cmake > 3.22`: If it's not installed, please [download and install CMake](https://cmake.org/download/) following the instructions appropriate for your system.
- `CUDA developer tools (nvcc)`: To check for its presence, enter `nvcc --version` in a terminal. If it's not installed, follow the [official NVIDIA guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to install it.

## Building
To build the framework, follow these steps:
1. Open a terminal and navigate to the project directory.
2. Create a build directory using `mkdir build && cd build`.
3. Configure the build by running `cmake ..`.
4. Build the project by running `make`.

## Flags
Additionally, there are flags which can change the compilation/operation of the framework in different ways. To use them, add them to your `make` command as such:
```shell
CPPFLAGS="<Flag 1> <Flag 2>" make rebuild
```
For example, to disable timers in the code, you would compile with the following:
```shell
CPPFLAGS="DISABLE_TIMING" make rebuild
```

Below is a list of those flags:
- `DISABLE_TIMING` Functionally disable all timers implemented using the built-in timing functionality.


