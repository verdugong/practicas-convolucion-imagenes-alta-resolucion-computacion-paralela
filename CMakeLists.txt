cmake_minimum_required(VERSION 3.24)
project(Practica1CUDA LANGUAGES CXX)

# OFF permite comprobar la referencia CPU en un equipo sin CUDA.
option(PRACTICA_ENABLE_CUDA "Compilar la implementacion CUDA" ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de compilacion" FORCE)
endif()

add_library(practica_cpu
    src/cpu/media_cpu.cpp
    src/cpu/nitidez_cpu.cpp
    src/cpu/laplaciano_cpu.cpp
    src/common/filtros.cpp
    src/common/imagen.cpp
    src/common/validacion.cpp)
target_include_directories(practica_cpu PUBLIC include)
if(MSVC)
    target_compile_options(practica_cpu PRIVATE /W4 /fp:strict)
else()
    target_compile_options(practica_cpu PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

if(PRACTICA_ENABLE_CUDA)
    # RTX 3070: arquitectura 86; se puede cambiar con -DCMAKE_CUDA_ARCHITECTURES=...
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "Arquitectura CUDA")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_library(practica_gpu src/gpu/filtros_gpu.cu src/gpu/media_gpu.cu
        src/gpu/nitidez_gpu.cu src/gpu/laplaciano_gpu.cu)
    target_include_directories(practica_gpu PUBLIC include)
    target_link_libraries(practica_gpu PUBLIC CUDA::cudart)
    # Evita fusionar multiplicacion y suma; facilita comparar con CPU.
    target_compile_options(practica_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()

add_executable(practica1 src/main.cpp)
target_link_libraries(practica1 PRIVATE practica_cpu)
if(PRACTICA_ENABLE_CUDA)
    target_link_libraries(practica1 PRIVATE practica_gpu)
    target_compile_definitions(practica1 PRIVATE PRACTICA_WITH_CUDA=1)
endif()

enable_testing()
add_executable(prueba_cpu tests/prueba_cpu.cpp)
target_link_libraries(prueba_cpu PRIVATE practica_cpu)
add_test(NAME referencia_cpu COMMAND prueba_cpu)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME referencia_cpu_python
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/prueba_cpu_python.py")
    add_test(NAME pipeline_regresion
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/prueba_pipeline.py"
                --exe "$<TARGET_FILE:practica1>")
    set_tests_properties(pipeline_regresion PROPERTIES RUN_SERIAL TRUE)
endif()

if(PRACTICA_ENABLE_CUDA)
    add_executable(prueba_gpu tests/prueba_gpu.cpp)
    target_link_libraries(prueba_gpu PRIVATE practica_cpu practica_gpu)
    add_test(NAME equivalencia_cpu_gpu COMMAND prueba_gpu)
endif()

# PyCUDA compila sus kernels al iniciar Python, no durante el build de CMake.
# El mismo interprete debe tener instaladas las dependencias de requirements.txt.
if(PRACTICA_ENABLE_CUDA)
    if(Python3_Interpreter_FOUND)
        add_test(NAME equivalencia_pycuda
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/prueba_pycuda.py"
                    --exe "$<TARGET_FILE:practica1>")
        set_tests_properties(equivalencia_cpu_gpu equivalencia_pycuda PROPERTIES RUN_SERIAL TRUE)
    else()
        message(WARNING "Python no encontrado: ejecutar tests/prueba_pycuda.py manualmente.")
    endif()
endif()
