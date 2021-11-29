# HaNNA (Halide Neural Network Acceleration)
Implement and investigate potential performance gain by implementing deep learning kernels with Halide
## To generate VS solution
```cmake "Visual Studio 16 2019" -Thost=x64 -A x64 -DCMAKE_PREFIX_PATH=Path\To\Halide\install\dir -S . -B build```
## Build with
```cmake --build .\build```
