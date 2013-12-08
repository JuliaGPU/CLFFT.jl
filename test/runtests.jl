using FactCheck 
using Base.Test

import OpenCL
const cl = OpenCL

import CLFFT
const clfft = CLFFT

macro throws_pred(ex) FactCheck.throws_pred(ex) end

facts("2D FFT Inplace") do
    const N = 512
    device, ctx, queue = cl.create_compute_context()

    X = ones(Complex64, (N, N))
    bufX = cl.Buffer(Complex64, ctx, :copy, hostbuf=X)

    p = clfft.Plan(Complex64, ctx, X)
    clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)
    R = cl.read(queue, bufX)
    
    err = norm(R - fft(X))
    @fact isapprox(err, zero(Float32)) => true
    Base.gc()
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

    X = ones(Complex64, N)
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
    @fact isapprox(norm(R - fft(X)), zero(Float32)) => true
    Base.gc()
end

facts("Example FFT Double") do
    const N = 1024
    device, ctx, queue = cl.create_compute_context()

    X = ones(Complex128, N)
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

        @fact isapprox(norm(R - fft(X)), zero(Float32)) => true
    catch err
        if err.desc == :CLFFT_DEVICE_NO_DOUBLE
            info("OpenCL.Device $device\ndoes not support double precision")
        else
            error("Error constructing Double Prec. Plan")
        end
    end
end
