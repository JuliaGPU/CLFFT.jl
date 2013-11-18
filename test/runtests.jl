using FactCheck 
using Base.Test

import OpenCL
const cl = OpenCL

import CLFFT
const clfft = CLFFT

macro throws_pred(ex) FactCheck.throws_pred(ex) end

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

facts("Example FFT") do
    const N = 1024
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

    clfft.bake(p, queue) 
    clfft.enqueue_transform(p, :forward, [queue], bufX, nothing)  
    cl.finish(queue)
    R = cl.read(queue, bufX)
    @fact isapprox(norm(R - fft(X)), zero(Float32)) => true
end
