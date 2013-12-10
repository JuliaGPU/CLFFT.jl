module CLFFT

import OpenCL
const cl = OpenCL

include("api.jl")
include("error.jl")

immutable CLFFTError
    code::Int
    desc::Symbol
    function CLFFTError(c::Integer)
        code = int(c)
        new(code, _clfft_error_codes[code])
    end
end

Base.show(io::IO, err::CLFFTError) = 
        Base.print(io, "CLFFTError(code=$(err.code), :$(err.desc))")

macro check(clfftFunc)
    quote
        local err::Cint
        err = $clfftFunc
        if err != api.CLFFT_SUCCESS
            throw(CLFFTError(err))
        end
    end
end

version() = begin
    major = cl.CL_uint[0]
    minor = cl.CL_uint[0]
    patch = cl.CL_uint[0]
    api.clfftGetVersion(major, minor, patch)
    return (int(major[1]), int(minor[1]), int(patch[1]))
end

# Module level library handle,
# when module is GC'd, the finalizer for SetupData is
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
    
    function Plan(plan::PlanHandle)
        p = new(plan)
        finalizer(p, x -> begin 
            if x.id != 0
                #TODO: this segfaults the julia interpreter
                #@check api.clfftDestroyPlan(x.id)
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
    if err != api.CLFFT_SUCCESS
        if ph[1] != 0
            @check api.clfftDestroyPlan(ph)
        end
        throw(CLFFTError(err))
    end

    if T <: clfftSingle
        @check api.clfftSetPlanPrecision(ph[1], api.CLFFT_SINGLE)
    else
        @check api.clfftSetPlanPrecision(ph[1], api.CLFFT_DOUBLE)
    end

    @assert ph[1] != 0
    Plan{T}(ph[1])
end

function Plan{T<:clfftNumber}(::Type{T}, ctx::cl.Context, 
                              input::StridedArray{T}, region)
    ndim = length(region)
    if ndim > 3
        throw(ArgumentError("Plans can have dimensions of 1, 2, or 3"))
    end
    if ndims(input) != ndim
        throw(ArgumentError("input array and region must have the same dimensionality"))
    end
    insize = size(input)
    lengths = Csize_t[0,0,0]
    for i in 1:ndim
        lengths[i] = insize[i]
    end
    ph = PlanHandle[0]
    err = api.clfftCreateDefaultPlan(ph, ctx.id, int32(ndim), lengths)
    if err != api.CLFFT_SUCCESS
        if ph[1] != 0
            @check api.clfftDestroyPlan(ph)
        end
        throw(CLFFTError(err))
    end
    if T <: clfftSingle
        @check api.clfftSetPlanPrecision(ph[1], api.CLFFT_SINGLE)
    else
        @check api.clfftSetPlanPrecision(ph[1], api.CLFFT_DOUBLE)
    end
    @assert ph[1] != 0
    plan = Plan{T}(ph[1])
    
    tdim = ndim
    tstrides = strides(input)
    # TODO : this works for dense arrays
    # need to test out for arbitrary regions
    tdistance = 0
    tbatchsize = 1
    
    set_result(plan, :inplace)
    set_layout(plan, :interleaved, :interleaved)
    set_instride(plan, tstrides)
    set_outstride(plan, tstrides)
    set_distance(plan, tdistance, tdistance)
    set_batchsize(plan, tbatchsize)

    return plan
end

Plan{T<:clfftNumber}(::Type{T}, ctx::cl.Context, input::StridedArray{T}) = 
        Plan(T, ctx, input, 1:ndims(input))

precision(p::Plan) = begin
    res = Cint[0]
    @check api.clfftGetPlanPrecision(p.id, res)
    if res[1] == 1
        return :single
    elseif res[1] == 2
        return :double
    elseif res[1] == 3
        return :single_fast
    elseif res[1] == 4
        return :double_fast
    else 
        error("undefined")
    end
end


set_precision(p::Plan, v::Symbol) = begin
    if v == :single
        @check api.clfftSetPlanPrecision(p.id, api.CLFFT_SINGLE)
    elseif v == :double
        @check api.clfftSetPlanPrecision(p.id, api.CLFFT_DOUBLE)
    elseif v == :single_fast
        @check api.clfftSetPlanPrecision(p.id, api.CLFFT_SINGLE_FAST)
    elseif v == :double_fast
        @check api.clfftSetPlanPrecision(p.id, api.CLFFT_DOUBLE_FAST)
    else
        error("unknown precision $v")
    end
end


layout(p::Plan) = begin
    i = Cint[0]
    o = Cint[0]
    @check api.clfftGetLayout(p.id, i, o)
    lout = x -> begin
        if x == 1
            return :interleaved
        elseif x == 2
            return :planar
        elseif x == 3
            return :hermitian_interleaved
        elseif x == 4
            return :hermitian_planar
        elseif x == 5 
            return :real
        else
            error("undefined")
        end
    end
    (lout(i[1]), lout(o[1]))
end


set_layout(p::Plan, in::Symbol, out::Symbol) = begin
    args = Cint[0, 0]
    if in == :interleaved
        args[1] = api.CLFFT_COMPLEX_INTERLEAVED
    elseif in == :planar
        args[1] = api.CLFFT_COMPLEX_PLANAR
    else
        throw(ArgumentError("in must be :interleaved or :planar"))
    end
    if in == :interleaved
        args[2] = api.CLFFT_COMPLEX_INTERLEAVED
    elseif in == :planar
        args[2] = api.CLFFT_COMPLEX_PLANAR
    else
        throw(ArgumentError("in must be :interleaved or :planar"))
    end
    @check api.clfftSetLayout(p.id, args[1], args[2])
end


result(p::Plan) = begin
    res = Cint[0]
    @check api.clfftGetResultLocation(p.id, res)
    if res[1] == 1
        return :inplace
    elseif res[1] == 2
        return :outofplace
    else
        error("undefined")
    end
end


set_result(p::Plan, v::Symbol) = begin
    if v == :inplace
        @check api.clfftSetResultLocation(p.id, api.CLFFT_INPLACE)
    elseif v == :outofplace
        @check api.clfftSetResultLocation(p.id, api.CLFFT_OUTOFPLACE)
    else
        throw(ArgumentError("set_result must be :inplace or :outofplace"))
    end
end


scaling_factor(p::Plan, dir::Symbol) = begin
    res = Cint[0]
    d::Cint
    if dir == :forward
        d = int32(-1)
    elseif dir == :backward
        d = int32(1)
    else
        error("undefined")
    end
    scale = Float32[0]
    @check api.clfftGetPlanScale(p.id, d, scale)
    return scale[1]
end


set_scaling_factor(p::Plan, dir::Symbol, f::FloatingPoint) = begin
    if dir == :forward
        d = int32(-1)
    elseif d == :backward
        d = int32(1)
    else
        error("undefined")
    end
    @check api.clfftsetPlanScale(p.id, d, float32(f))
end


set_batchsize(p::Plan, n::Integer) = begin 
    @assert n > 0
    @check api.clfftSetPlanBatchSize(p.id, convert(Csize_t, n))
end


batchsize(p::Plan) = begin
    res = Csize_t[0]
    @check api.clfftGetPlanBatchSize(p.id, res)
    return int(res[1])
end


set_dim(p::Plan, d::Integer) = begin
    @assert d > 0 && d <= 3
    @check api.clfftSetPlanDim(p.id, convert(Csize_t, d))
end


dim(p::Plan) = begin
    res  = Int32[0]
    size = Csize_t[0]
    @check api.clfftGetPlanDim(p.id, res, size)
    return int(res[1])
end


set_lengths(p::Plan, dims::Dims) = begin
    ndim = length(dims)
    @assert ndim <= 3
    nd = Array(Csize_t, ndim)
    for (i, d) in enumerate(dims)
        nd[i] = d
    end
    @check api.clfftSetPlanLength(p.id, int32(ndim), nd)
end


lengths(p::Plan) = begin
    d = dim(p)
    res = Array(Csize_t, d)
    @check api.clfftGetPlanLength(p.id, int32(d), res)
    return int(res)
end


instride(p::Plan) = begin
    d = dim(p)
    res = Array(Csize_t, d)
    @check api.clfftGetPlanInStride(p.id, int32(d), res)
    return int(res)
end


set_instride(p::Plan, instrides) = begin
    d = length(instrides)
    @assert d == dim(p)
    strides = Csize_t[int(s) for s in instrides]
    @check api.clfftSetPlanInStride(p.id, int32(d), strides)
end


outstride(p::Plan) = begin
    d = dim(p)
    res = Array(Csize_t, d)
    @check api.clfftGetPlanOutStride(p.id, int32(d), res)
    return int(res)
end


set_outstride(p::Plan, outstrides) = begin
    d = length(outstrides)
    @assert d == dim(p)
    strides = Csize_t[int(s) for s in outstrides]
    api.clfftSetPlanInStride(p.id, int32(d), strides)
end


distance(p::Plan) = begin
    indist = Csize_t[0]
    odist  = Csize_t[0]
    @check api.clfftGetPlanDistance(p.id, indist, odist)
    return (indist[1], odist[1])
end


set_distance(p::Plan, indist::Integer, odist::Integer) = begin 
    @assert indist >= 0 && odist >= 0
    i = uint(indist)
    o = uint(odist)
    @check api.clfftSetPlanDistance(p.id, i, o)
end


transpose_result(p::Plan) = begin
    res = Cint[0]
    @check api.clfftGetPlanTransposeResult(p.id, res)
    if res[1] == api.CLFFT_NOTRANSPOSE
        return false
    elseif res[1] == api.CLFFT_TRANSPOSED
        return true
    else
        error("undefined")
    end
end


set_transpose_result(p::Plan, transpose::Bool) = begin
    if transpose
        @check api.clfftSetTransposeResult(p.id,
                cl.CLFFT_TRANSPOSED)
    else
        @check api.clfftSetTransposeResult(p.id,
                cl.CLFFT_NOTRANSPOSE)
    end
end


tmp_buffer_size(p::Plan) = begin
    res = Csize_t[0]
    @check api.clfftGetTmpBufSize(p.id, res)
    return int(res[1])
end


context(p::Plan) = begin
    res = Array(cl.CL_context)
    @check api.clfftGetPlanContext(p.id, res)
    return cl.Context(res[1])
end


bake(p::Plan, qs::Vector{cl.CmdQueue}) = begin
    nqueues = length(qs)
    q_ids = [q.id for q in qs]
    # TODO: callback
    @check api.clfftBakePlan(p.id, nqueues, q_ids, C_NULL, C_NULL)
end
bake(p::Plan, q::cl.CmdQueue) = bake(p, [q])


function enqueue_transform{T<:clfftNumber}(p::Plan,
                                           dir::Symbol,
                                           qs::Vector{cl.CmdQueue},
                                           in::cl.Buffer{T},
                                           out::Union(Nothing,cl.Buffer{T});
                                           wait_for::Union(Nothing,Vector{cl.Event})=nothing,
                                           tmp::Union(Nothing,cl.Buffer{T})=nothing)
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
    @check api.clfftEnqueueTransform(
                              p.id,
                              dir == :forward ? api.CLFFT_FORWARD : api.CLFFT_BACKWARD,
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
