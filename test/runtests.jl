using FactCheck 
using Base.Test

import OpenCL
const cl = OpenCL

import CLFFT
const clfft = CLFFT

macro throws_pred(ex) FactCheck.throws_pred(ex) end

const TOLERANCE = 1e-3

function allclose{T}(x::AbstractArray{T}, y::AbstractArray{T}; rtol=1e-5, atol=1e-8)
    @assert length(x) == length(y)
    @inbounds begin 
        for i in length(x)
            xx, yy = x[i], y[i]
            if !(isapprox(xx, yy; rtol=rtol, atol=atol))
                return false
            end
        end
    end
    return true
end

#Note: these functions have been adapted from the clFFT tests (buffer.h)
function floats_about_equal{T<:clfft.clfftNumber}(a::T, b::T)
    tol  = convert(T, TOLERANCE)
    comp = 1e-5
    if T <: Real
        if abs(a) < comp && abs(b) < comp
            return true
        end
        return abs(a - b) > abs(a * tol) ? false : true
    else
        return (floats_about_equal(real(a), real(b)) &&
                floats_about_equal(imag(b), imag(b)))
    end
end

function allclose_clfft{T<:clfft.clfftNumber}(x::AbstractArray{T}, y::AbstractArray{T})
    @assert length(x) == length(y)
    @inbounds begin
        for i in length(x)
            xx, yy = x[i], y[i]
            if !(floats_about_equal(xx, yy))
                return false
            end
        end
    end
    return true
end

facts("2D FFT Inplace") do
    const N = 1024
    device, ctx, queue = cl.create_compute_context()
    X = rand(Complex64, (N, N))
    p = clfft.Plan(Complex64, ctx, X)
    bufX = cl.Buffer(Complex64, ctx, :copy, hostbuf=X)
    clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)
    R = reshape(cl.read(queue, bufX), size(X))
    @fact allclose(R, fft(X); rtol=1e-2, atol=1e-3) => true
    @fact allclose_clfft(R, fft(X)) => true
end

facts("3D FFT Inplace") do
    const N = 256
    device, ctx, queue = cl.create_compute_context()
    X = rand(Complex64, (N, N, N))
    p = clfft.Plan(Complex64, ctx, X)
    bufX = cl.Buffer(Complex64, ctx, :copy, hostbuf=X)
    clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)
    R = reshape(cl.read(queue, bufX), size(X))
    @fact allclose(R, fft(X); rtol=1e-2, atol=1e-3) => true
    @fact allclose_clfft(R, fft(X)) => true
end

facts("Version") do 
    @fact isa(CLFFT.version(), NTuple{3,Int}) => true
    @fact clfft.version()[1] >= 2 => true
    @fact clfft.version()[2] >= 1 => true
    @fact clfft.version()[3] >= 0 => true
end

facts("Plan") do
    context("Constructor") do
        ctx = cl.create_some_context()
        @fact @throws_pred(clfft.Plan(Complex64, ctx, (10, 10))) => (false, "no error")
    end
end

facts("Example FFT Single") do
    const N = 512
    _, ctx, queue = cl.create_compute_context()

    X = rand(Complex64, N)
    bufX = cl.Buffer(Complex64, ctx, :copy, hostbuf=X)

    p = clfft.Plan(Complex64, ctx, size(X))
    clfft.set_layout(p, :interleaved, :interleaved)
    clfft.set_result(p, :inplace)
    
    @fact clfft.context(p) => ctx
    @fact clfft.precision(p) => :single
    @fact clfft.layout(p) => (:interleaved, :interleaved)
    @fact clfft.result(p) => :inplace
    @fact clfft.dim(p) => 1
    @fact length(clfft.lengths(p)) => 1
    @fact clfft.lengths(p)[1] => length(X)
    @fact clfft.transpose_result(p) => false

    @fact clfft.scaling_factor(p, :forward) => float32(1.0)
    @fact clfft.scaling_factor(p, :backward) => float32(1.0 / length(X))
    @fact clfft.batchsize(p) => 1

    #clfft.bake(p, queue) 
    clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)  
    # read is blocking (waits on pending event for result)
    R = cl.read(queue, bufX)
    @fact allclose(R, fft(X); rtol=1e-2, atol=1e-3) => true
    @fact allclose_clfft(R, fft(X)) => true
end

facts("Example FFT Double") do
    const N = 1024
    device, ctx, queue = cl.create_compute_context()

    X = rand(Complex128, N)
    bufX = cl.Buffer(Complex128, ctx, :copy, hostbuf=X)
    try  
        p = clfft.Plan(Complex128, ctx, size(X))
        clfft.set_layout(p, :interleaved, :interleaved)
        clfft.set_result(p, :inplace)
        
        @fact clfft.context(p) => ctx
        @fact clfft.precision(p) => :double
        @fact clfft.layout(p) => (:interleaved, :interleaved)
        @fact clfft.result(p) => :inplace
        @fact clfft.dim(p) => 1
        @fact length(clfft.lengths(p)) => 1
        @fact clfft.lengths(p)[1] => length(X)
        @fact clfft.transpose_result(p) => false

        @fact clfft.scaling_factor(p, :forward) => float32(1.0)
        @fact clfft.scaling_factor(p, :backward) => float32(1.0 / length(X))
        @fact clfft.batchsize(p) => 1

#        clfft.bake(p, queue) 
        clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)  
        R = cl.read(queue, bufX)
        @fact allclose(R, fft(X); rtol=1e-2, atol=1e-3) => true
        @fact allclose_clfft(R, fft(X)) => true
    catch err
        if err.desc == :CLFFT_DEVICE_NO_DOUBLE
            info("OpenCL.Device $device\ndoes not support double precision")
        else
            error("Error constructing Double Prec. Plan")
        end
    end
end
