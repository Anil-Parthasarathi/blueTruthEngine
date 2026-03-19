# RedTruthEngine

A CUDA + OpenGL rendering engine. CUDA writes pixels into a Pixel Buffer Object; OpenGL displays it as a fullscreen quad.

Aiming to create a fully loaded path tracer with support for neural rendering and cool physical simulation features!

## Requirements

- Windows 10/11
- CUDA Toolkit 12.x
- Visual Studio 2022 (includes CMake 3.28)
- NVIDIA GPU (SM 8.0+)

## Build

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

# Configure (downloads GLFW + GLM automatically)
& $cmake -S . -B build -G "Visual Studio 17 2022" -A x64

# Build
& $cmake --build build --config Release
```

## Run

```powershell
.\build\Release\RedTruthEngine.exe
```

Press **Escape** or close the window to exit.
