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
                #TODO: this segfaults the julia interpreter
                #api.clfftDestroyPlan(x.id)
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

    if T <: clfftSingle
        api.clfftSetPlanPrecision(ph[1], api.clfftPrecision.SINGLE_FAST)
    else
        api.clfftSetPlanPrecision(ph[1], api.clfftPrecision.DOUBLE_FAST)
    end

    @assert ph[1] != 0
    Plan{T}(ph[1], sz)
end

set_precision(p::Plan, v::Symbol) = begin
    if v == :single
        api.clfftSetPlanPrecision(p.id, api.clfftPrecision.SINGLE)
    elseif v == :double
        api.clfftSetPlanPrecision(p.id, api.clfftPrecision.DOUBLE)
    elseif v == :single_fast
        api.clfftSetPlanPrecision(p.id, api.clfftPrecision.SINGLE_FAST)
    elseif v == :double_fast
        api.clfftSetPlanPrecision(p.id, api.clfftPrecision.DOUBLE_FAST)
    else
        error("unknown precision $v")
    end
end

set_layout(p::Plan, in::Symbol, out::Symbol) = begin
    args = Cint[0, 0]
    if in == :interleaved
        args[1] = api.clfftLayout.COMPLEX_INTERLEAVED
    elseif in == :planar
        args[1] = api.clfftLayout.COMPLEX_PLANAR
    else
        throw(ArgumentError("in must be :interleaved or :planar"))
    end
    if in == :interleaved
        args[2] = api.clfftLayout.COMPLEX_INTERLEAVED
    elseif in == :planar
        args[2] = api.clfftLayout.COMPLEX_PLANAR
    else
        throw(ArgumentError("in must be :interleaved or :planar"))
    end
    api.clfftSetLayout(p.id, args[1], args[2])
end

set_result(p::Plan, v::Symbol) = begin
    if v == :inplace
        api.clfftSetResultLocation(p.id, api.clfftResultLocation.INPLACE)
    elseif v == :outofplace
        api.clfftSetResultLocation(p.id, api.clfftResultLocation.OUTOFPLACE)
    else
        throw(ArgumentError("set_result must be :inplace or :outofplace"))
    end
end

bake(p::Plan, qs::Vector{cl.CmdQueue}) = begin
    nqueues = length(qs)
    q_ids = [q.id for q in qs]
    # TODO: callback
    api.clfftBakePlan(p.id, uint32(nqueues), q_ids, C_NULL, C_NULL)
end
bake(p::Plan, q::cl.CmdQueue) = bake(p, [q])

function enqueue_transform{T<:clfftNumber}(p::Plan,
                                           dir::Symbol,
                                           qs::Vector{cl.CmdQueue},
                                           in::cl.Buffer{T},
                                           out::Union(Nothing,cl.Buffer{T});
                                           wait_for::Union(Nothing,Vector{cl.Event})=nothing,
                                           tmp::Union(Nothing,cl.Buffer{T})=nothing)
    FORWARD  = api.clfftDirection.FORWARD
    BACKWARD = api.clfftDirection.BACKWARD
    if dir != :forward && dir != :backward
        throw(ArgumentError("Unknown direction $dir"))
    end
    
    q_ids = [q.id for q in qs]
    in_buff_ids  = [in.id]
    
    out_buff_ids = C_NULL
    if out != nothing
        out_buff_ids = [out.id]
    end 

    nevts = 0
    evt_ids = C_NULL 
    if wait_for != nothing
        nevts = length(wait_for)
        evt_ids = [evt.id for evt in wait_for]
    end

    nqueues = length(q_ids)
    out_evts = Array(cl.CL_event, nqueues)
    
    tmp_buffer = C_NULL
    if tmp != nothing
        tmp_buff_id = [tmp.id]
    end

    api.clfftEnqueueTransform(p.id,
                              dir == :forward ? FORWARD : BACKWARD,
                              uint32(nqueues),
                              q_ids,
                              uint32(nevts),
                              evt_ids,
                              out_evts,
                              in_buff_ids,
                              out_buff_ids,
                              tmp_buffer)

    return [cl.Event(e_id) for e_id in out_evts]
end

        
end # module
