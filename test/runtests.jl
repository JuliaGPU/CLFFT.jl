using FactCheck 
using Base.Test

import OpenCL
const cl = OpenCL

import CLFFT

macro throws_pred(ex) FactCheck.throws_pred(ex) end

facts("Version") do 
    @fact isa(CLFFT.version(), NTuple{3,Int}) => true
    @fact CLFFT.version()[1] >= 2 => true
    @fact CLFFT.version()[2] >= 1 => true
    @fact CLFFT.version()[3] >= 0 => true
end

facts("Plan") do
    context("Constructor") do
        ctx = cl.create_some_context()
        @fact @throws_pred(CLFFT.Plan(Complex64, ctx, (10, 10))) => (false, "no error")
        p = CLFFT.Plan(Complex64, ctx, (10, 10))
    end
end

