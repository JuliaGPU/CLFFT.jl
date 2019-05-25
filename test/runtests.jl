using Test
using Primes
using FFTW

import OpenCL.cl

import CLFFT
const clfft = CLFFT

#macro throws_pred(ex) FactCheck.throws_pred(ex) end

const TOLERANCE = 1e-3

function allclose(x::AbstractArray{T}, y::AbstractArray{T}; rtol=1e-5, atol=1e-8) where {T}
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
function floats_about_equal(a::T, b::T) where {T<:clfft.clfftNumber}
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

function allclose_clfft(x::AbstractArray{T}, y::AbstractArray{T}) where {T<:clfft.clfftNumber}
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


@testset "CLFFT.jl" begin
    @testset "Version" begin
        @test isa(CLFFT.version(), VersionNumber)
        v = clfft.version()
        @test v.major >= 2
        @test v.minor >= 1
        @test v.patch >= 0
    end

    @testset "Example FFT Single" begin
        for N in (2^8,)# 3^7, 5^6]
            X = rand(ComplexF32, N)
            fftw_X = fft(X)
            for device in cl.devices()
                ctx = cl.Context(device)
                queue = cl.CmdQueue(ctx)
                bufX = cl.Buffer(ComplexF32, ctx, :copy, hostbuf=X)
                p = clfft.Plan(ComplexF32, ctx, size(X))
                clfft.set_layout!(p, :interleaved, :interleaved)
                clfft.set_result!(p, :inplace)
                clfft.bake!(p, queue)

                @test clfft.context(p) == ctx
                @test clfft.precision(p) == :single
                @test clfft.layout(p) == (:interleaved, :interleaved)
                @test clfft.result(p) == :inplace
                @test clfft.dim(p) == 1
                @test length(clfft.lengths(p)) == 1
                @test clfft.lengths(p)[1] == length(X)
                @test clfft.transpose_result(p) == false

                @test clfft.scaling_factor(p, :forward) == Float32(1.0)
                @test clfft.scaling_factor(p, :backward) == Float32(1.0 / length(X))
                @test clfft.batchsize(p) == 1

                clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)
                # read is blocking (waits on pending event for result)
                R = cl.read(queue, bufX)
                @test allclose(R, fftw_X; rtol=1e-2, atol=1e-3)
                @test allclose_clfft(R, fftw_X)
            end
        end
    end

    @testset "Example FFT Double" begin
        for N in (2^7,)# 3^6, 5^5]
            X = rand(ComplexF64, N)
            fftw_X = fft(X)
            for device in cl.devices()
                ctx = cl.Context(device)
                queue = cl.CmdQueue(ctx)
                bufX = cl.Buffer(ComplexF64, ctx, :copy, hostbuf=X)
                p = clfft.Plan(ComplexF64, ctx, size(X))
                clfft.set_layout!(p, :interleaved, :interleaved)
                clfft.set_result!(p, :inplace)
                clfft.bake!(p, queue)

                @test clfft.context(p) == ctx
                @test clfft.precision(p) == :double
                @test clfft.layout(p) == (:interleaved, :interleaved)
                @test clfft.result(p) == :inplace
                @test clfft.dim(p) == 1
                @test length(clfft.lengths(p)) == 1
                @test clfft.lengths(p)[1] == length(X)
                @test clfft.transpose_result(p) == false

                @test clfft.scaling_factor(p, :forward) == Float32(1.0)
                @test clfft.scaling_factor(p, :backward) == Float32(1.0 / length(X))
                @test clfft.batchsize(p) == 1

                clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)
                R = cl.read(queue, bufX)
                @test allclose(R, fftw_X; rtol=1e-2, atol=1e-3)
                @test allclose_clfft(R, fftw_X)

            end
        end
    end

    @testset "2D FFT Inplace" begin
        transform_sizes = (2^6,)#3^4, 5^3
        for N in transform_sizes
            for M in transform_sizes
                X = rand(ComplexF32, (N, M))
                fftw_X = fft(X)
                for device in cl.devices()
                    ctx = cl.Context(device)
                    queue = cl.CmdQueue(ctx)
                    p = clfft.Plan(ComplexF32, ctx, X)
                    clfft.bake!(p, queue)
                    bufX = cl.Buffer(ComplexF32, ctx, :copy, hostbuf=X)
                    clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)
                    R = reshape(cl.read(queue, bufX), size(X))
                    @test allclose(R, fftw_X; rtol=1e-2, atol=1e-3)
                    @test allclose_clfft(R, fftw_X)
                    Base.gc()
                end
            end
        end
    end

    @testset "3D FFT Inplace" begin
        N = 64
        X = rand(ComplexF32, (N, N, N))
        fftw_X = fft(X)
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            p = clfft.Plan(ComplexF32, ctx, X)
            clfft.bake!(p, queue)
            bufX = cl.Buffer(ComplexF32, ctx, :copy, hostbuf=X)
            clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)
            R = reshape(cl.read(queue, bufX), size(X))
            @test allclose(R, fftw_X; rtol=1e-2, atol=1e-3)
            @test allclose_clfft(R, fftw_X)
            Base.gc()
        end
    end
end
