# Blue Truth Engine

A CUDA + OpenGL rendering engine. CUDA writes pixels into a Pixel Buffer Object; OpenGL displays it as a fullscreen quad.

<div  align="center" > <img width="800" height="600" alt="screenshot_0000" src="https://github.com/user-attachments/assets/c2b431d4-4758-436a-b831-6ab988bb66f8" /> </div>

<img width="1728" height="1118" alt="completePokemonShowcase" src="https://github.com/user-attachments/assets/2f92fd1b-bf7c-4129-8f7f-e63d92cd9e2b" />

<img width="1440" height="1440" alt="pietaDrama8192-2hr" src="https://github.com/user-attachments/assets/5b6abb27-3aec-40fc-8d7a-fd72c563e693" />

<img width="1280" height="720" alt="anilRenderWhiteFloor" src="https://github.com/user-attachments/assets/c96d0eff-54d5-48fb-8ef2-fbcf77a7a775" />



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
.\build\Release\BlueTruthEngine.exe
```

Press **Escape** or close the window to exit.
