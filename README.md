# CLFFT

Julia bindings to clFFT library.

## Example

```julia
import OpenCL
import CLFFT
import FFTW
using LinearAlgebra

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

@assert isapprox(norm(result - FFTW.fft(X)), zero(Float32))
```
