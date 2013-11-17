module CLFFT

import OpenCL
const cl = OpenCL

include("api.jl")

version() = begin
    major = cl.CL_uint[0]
    minor = cl.CL_uint[0]
    patch = cl.CL_uint[0]
    api.clfftGetVersion(major, minor, patch)
    return (int(major[1]), int(minor[1]), int(patch[1]))
end

# Module level library handle,
# when module is GC'd, the finalizer on SetupData is
# called to teardown the library state
const __handle = begin
    v = version()
    api.SetupData(v[1], v[2], v[3], 0)
end

# clFFT floating-point types:
typealias clfftNumber Union(Float64,Float32,Complex128,Complex64)
typealias clfftReal Union(Float64,Float32)
typealias clfftComplex Union(Complex128,Complex64)
typealias clfftDouble Union(Float64,Complex128)
typealias clfftSingle Union(Float32,Complex64)
typealias clfftTypeDouble Union(Type{Float64},Type{Complex128})
typealias clfftTypeSingle Union(Type{Float32},Type{Complex64})

typealias PlanHandle Csize_t

type Plan{T<:clfftNumber}
    id::PlanHandle
    sz::Dims # size of array on which plan operates (Int tuple)
    istride::Dims # strides of input
    
    function Plan(plan::PlanHandle, sz::Dims)
        println("constructing plan $(plan)")
        p = new(plan,sz,sz)
        finalizer(p, x -> begin 
            if x.id != 0
                println("destroying plan $(x.id)")
                api.clfftDestroyPlan(x.id)
            end
            x.id = 0
        end)
        return p
    end
end

function Plan{T<:clfftNumber}(::Type{T}, ctx::cl.Context, sz::Dims)
    if length(sz) > 3
        throw(ArgumentError("Plans can have dimensions of 1,2 or 3"))
    end
    ndim = length(sz) 
    lengths = Csize_t[0, 0, 0]
    for i in 1:ndim
        lengths[i] = sz[i]
    end
    ph = PlanHandle[0]
    err = api.clfftCreateDefaultPlan(ph, ctx.id, 
                                     int32(ndim), lengths)
    if err != api.clfftStatus.SUCCESS
        if ph[1] != 0
            api.clfftDestroyPlan(ph)
        end
        error("Error creating Plan: code $err")
    end
    @assert ph[1] != 0
    Plan{T}(ph[1], sz)
end


end # module
