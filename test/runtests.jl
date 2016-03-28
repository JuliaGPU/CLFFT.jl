using FactCheck
using Base.Test

import OpenCL
const cl = OpenCL

import CLFFT
const clfft = CLFFT

#macro throws_pred(ex) FactCheck.throws_pred(ex) end

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


facts("Version") do
    @fact isa(CLFFT.version(), VersionNumber) --> true
    v = clfft.version()
    @fact v.major >= 2 --> true
    @fact v.minor >= 1 --> true
    @fact v.patch >= 0 --> true
end

facts("Plan") do
    context("Constructor") do
        ctx = cl.create_some_context()
        @fact clfft.Plan(Complex64, ctx, (10, 10)) --> not(nothing)
        # Plan's throw error on non-muliple 2,3,or 5 dims
        for x in [2,3,5]
            @fact (clfft.Plan(Complex64, ctx, (x^3,))) --> not(nothing)
            @fact_throws (clfft.Plan(Complex64, ctx, (7^3,)))
            for y in [2,3,5]
                @fact (clfft.Plan(Complex64, ctx, (x^3, y^3))) --> not(nothing)
                @fact_throws (clfft.Plan(Complex64, ctx, (7^3, y^3)))
                for z in [2,3,5]
                @fact (clfft.Plan(Complex64, ctx, (x^3, y^3, z^3))) --> not(nothing)
                @fact_throws (clfft.Plan(Complex64, ctx, (x^3, 7^3, z^3)))
                end
            end
        end
        # FFT only for 1,2 or 3 dim
        @fact_throws (clfft.Plan(Complex64, ctx, (2^2, 2^2, 2^2, 2^2)))
        @fact_throws (clfft.Plan(Complex64, ctx, ()))
        Base.gc()
    end
end

facts("Example FFT Single") do
    for N in [2^8,]# 3^7, 5^6]
        @show N
        X = rand(Complex64, N)
        fftw_X = fft(X)
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            bufX = cl.Buffer(Complex64, ctx, :copy, hostbuf=X)
            p = clfft.Plan(Complex64, ctx, size(X))
            clfft.set_layout!(p, :interleaved, :interleaved)
            clfft.set_result!(p, :inplace)
            clfft.bake!(p, queue)

            @fact clfft.context(p) --> ctx
            @fact clfft.precision(p) --> :single
            @fact clfft.layout(p) --> (:interleaved, :interleaved)
            @fact clfft.result(p) --> :inplace
            @fact clfft.dim(p) --> 1
            @fact length(clfft.lengths(p)) --> 1
            @fact clfft.lengths(p)[1] --> length(X)
            @fact clfft.transpose_result(p) --> false

            @fact clfft.scaling_factor(p, :forward) --> Float32(1.0)
            @fact clfft.scaling_factor(p, :backward) --> Float32(1.0 / length(X))
            @fact clfft.batchsize(p) --> 1

            clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)
            # read is blocking (waits on pending event for result)
            R = cl.read(queue, bufX)
            @fact allclose(R, fftw_X; rtol=1e-2, atol=1e-3) --> true
            @fact allclose_clfft(R, fftw_X) --> true
        end
    end
end

facts("Example FFT Double") do
    for N in [2^7,]# 3^6, 5^5]
        @show N
        X = rand(Complex128, N)
        fftw_X = fft(X)
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            try
                bufX = cl.Buffer(Complex128, ctx, :copy, hostbuf=X)
                p = clfft.Plan(Complex128, ctx, size(X))
                clfft.set_layout!(p, :interleaved, :interleaved)
                clfft.set_result!(p, :inplace)
                clfft.bake!(p, queue)

                @fact clfft.context(p) --> ctx
                @fact clfft.precision(p) --> :double
                @fact clfft.layout(p) --> (:interleaved, :interleaved)
                @fact clfft.result(p) --> :inplace
                @fact clfft.dim(p) --> 1
                @fact length(clfft.lengths(p)) --> 1
                @fact clfft.lengths(p)[1] --> length(X)
                @fact clfft.transpose_result(p) --> false

                @fact clfft.scaling_factor(p, :forward) --> Float32(1.0)
                @fact clfft.scaling_factor(p, :backward) --> Float32(1.0 / length(X))
                @fact clfft.batchsize(p) --> 1

                clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)
                R = cl.read(queue, bufX)
                @fact allclose(R, fftw_X; rtol=1e-2, atol=1e-3) --> true
                @fact allclose_clfft(R, fftw_X) --> true
            catch err
                if isa(err, clfft.CLFFTError)
                    if err.desc == :CLFFT_DEVICE_NO_DOUBLE
                        info("OpenCL.Device $device\ndoes not support double precision")
                    end
                else
                    throw(err)
                end
            end
        end
    end
end

facts("2D FFT Inplace") do
    transform_sizes = [2^6,]#3^4, 5^3]
    for N in transform_sizes
        for M in transform_sizes
            @show (N, M)
            X = rand(Complex64, (N, M))
            fftw_X = fft(X)
            for device in cl.devices()
                ctx = cl.Context(device)
                queue = cl.CmdQueue(ctx)
                p = clfft.Plan(Complex64, ctx, X)
                clfft.bake!(p, queue)
                bufX = cl.Buffer(Complex64, ctx, :copy, hostbuf=X)
                clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)
                R = reshape(cl.read(queue, bufX), size(X))
                @fact allclose(R, fftw_X; rtol=1e-2, atol=1e-3) --> true
                @fact allclose_clfft(R, fftw_X) --> true
                Base.gc()
            end
        end
    end
end

facts("3D FFT Inplace") do
    const N = 64
    X = rand(Complex64, (N, N, N))
    fftw_X = fft(X)
    for device in cl.devices()
        ctx = cl.Context(device)
        queue = cl.CmdQueue(ctx)
        p = clfft.Plan(Complex64, ctx, X)
        clfft.bake!(p, queue)
        bufX = cl.Buffer(Complex64, ctx, :copy, hostbuf=X)
        clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)
        R = reshape(cl.read(queue, bufX), size(X))
        @fact allclose(R, fftw_X; rtol=1e-2, atol=1e-3) --> true
        @fact allclose_clfft(R, fftw_X) --> true
        Base.gc()
    end
end

exitstatus()
