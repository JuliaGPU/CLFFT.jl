# CLFFT

[![Build Status](https://travis-ci.org/JuliaGPU/CLFFT.jl.png)](https://travis-ci.org/JuliaGPU/CLFFT.jl)
[![Coverage Status](https://img.shields.io/coveralls/JuliaGPU/CLFFT.jl.svg)](https://coveralls.io/r/JuliaGPU/CLFFT.jl)

Julia bindings to AMD's clFFT library

## Example

```julia
import OpenCL
import CLFFT

const cl = OpenCL
const clfft = CLFFT

_, ctx, queue = cl.create_compute_context()

N = 100
X = ones(Complex64, N)
bufX = cl.Buffer(Complex64, ctx, :copy, hostbuf=X)

p = clfft.Plan(Complex64, ctx, size(X))
clfft.set_layout!(p, :interleaved, :interleaved)
clfft.set_result!(p, :inplace)
clfft.bake!(p, queue)

clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)  
result = cl.read(queue, bufX)

@assert isapprox(norm(result - fft(X)), zero(Float32))
```
