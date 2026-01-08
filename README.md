# HIPSmith

## About

HIPSmith is a random generator of AMD HIP programs, built on top of [CSmith](https://github.com/csmith-project/csmith). Its primary purpose is to find bugs in HIP compilers (like AMD's ROCm/HIP toolchain) using differential testing as the test oracle.

Like its parent project CSmith, HIPSmith outputs programs free of undefined behaviors, but tailored for the heterogeneous computing environment of GPUs and CPUs.

## Install HIPSmith

You can build HIPSmith from source. The following commands assume a standard Linux environment (e.g., Ubuntu).

### Prerequisites
* Git
* C++ Compiler (GCC or Clang)
* CMake (Version 2.8.12 or higher)
* M4 (Required for generating safe math macros)
* Python 3 (For running the test script)

### Build Instructions

1.  **Create a build directory (Out-of-source build recommended):**
    ```bash
    mkdir build
    cd build
    ```

2.  **Configure and Build:**
    ```bash
    # Configure the project
    cmake ..

    # Compile
    make
    ```
    *Note: `cmake ..` will automatically copy `csmith.h`, `HIPSmith.h`, and generate `safe_math_macros.h` into your build directory.*

## Use HIPSmith

The primary way to use HIPSmith for differential testing is via the provided Python script, `hip_test.py`. This script automates the process of generating random HIP programs, compiling them, and executing them to find inconsistencies.

### Running Automated Tests

To start the testing loop:

```bash
python3 hip_test.py