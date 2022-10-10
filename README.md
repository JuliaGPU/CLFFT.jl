# CLFFT

[gitlab-img]: https://gitlab.com/JuliaGPU/CLFFT.jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/CLFFT.jl/pipelines

Julia bindings to clFFT library

**WARNING:** Some functions in this package FAIL in tests and aren't considered stable. Please, familiarize yourself
with [tests](https://github.com/JuliaGPU/CLFFT.jl/blob/master/test/runtests.jl) before using the package. 

## Example

```julia
import OpenCL
import CLFFT
import FFTW
import LinearAlgebra

const cl = OpenCL.cl
const clfft = CLFFT

_, ctx, queue = cl.create_compute_context()

N = 100
X = ones(ComplexF64, N)
bufX = cl.Buffer(ComplexF64, ctx, :copy, hostbuf=X)

p = clfft.Plan(ComplexF64, ctx, size(X))
clfft.set_layout!(p, :interleaved, :interleaved)
clfft.set_result!(p, :inplace)
clfft.bake!(p, queue)

clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)  
result = cl.read(queue, bufX)

@assert isapprox(LinearAlgebra.norm(result - FFTW.fft(X)), zero(Float32))
```
