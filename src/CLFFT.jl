module CLFFT

import OpenCL
const cl = OpenCL

include("api.jl")
include("error.jl")

const SP_MAX_LEN = 1 << 24
const DP_MAX_LEN = 1 << 22

immutable CLFFTError
    code::Int
    desc::Symbol
    function CLFFTError(c::Integer)
        code = Int(c)
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
        err
    end
end

version() = begin
    major = cl.CL_uint[0]
    minor = cl.CL_uint[0]
    patch = cl.CL_uint[0]
    api.clfftGetVersion(major, minor, patch)
    return VersionNumber(Int(major[1]), Int(minor[1]), Int(patch[1]))
end

# Module level library handle,
# when module is GC'd, the finalizer for SetupData is
# called to teardown the library state
const __handle = begin
    v = version()
    api.SetupData(v.major, v.minor, v.patch, 0)
end

# clFFT floating-point types:
typealias clfftNumber Union{Float64,Float32,Complex128,Complex64}
typealias clfftReal Union{Float64,Float32}
typealias clfftComplex Union{Complex128,Complex64}
typealias clfftDouble Union{Float64,Complex128}
typealias clfftSingle Union{Float32,Complex64}
typealias clfftTypeDouble Union{Type{Float64},Type{Complex128}}
typealias clfftTypeSingle Union{Type{Float32},Type{Complex64}}

typealias PlanHandle Csize_t


type Plan{T<:clfftNumber}
    # boxed handle (most api functions need address, setup/teardown need pointer)
    id::Array{PlanHandle,1}

    function Plan(plan::Array{PlanHandle,1})
        p = new(plan)
        finalizer(p, x -> begin
            if x.id[1] != C_NULL
                @check api.clfftDestroyPlan(x.id)
            end
            x.id[1] = C_NULL
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
    total_length = 1
    for i in 1:ndim
        s = sz[i]
        if mod(s, 2) == 0 || mod(s, 3) == 0 || mod(s, 5) == 0
            lengths[i] = s
            total_length *= s
        else
            throw(ArgumentError("""Plans can only have dims that are
                                   powers of 2, 3, or 5"""))
        end
    end
    if T <: clfftSingle
        if total_length > SP_MAX_LEN
            throw(ArgumentError("""CLFFT supports single precision transform
                                   lengths up to $(SP_MAX_LEN)"""))
        end
    else
        if total_length > DP_MAX_LEN
            throw(ArgumentError("""clFFT supports double precision transform
                                 lengths up to $(DP_MAX_LEN)"""))
        end
    end
    ph = PlanHandle[0]
    err = api.clfftCreateDefaultPlan(ph, ctx.id,
                                     Int32(ndim), lengths)
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
    return Plan{T}(ph)
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
    total_length = 1
    for i in 1:ndim
        s = insize[i]
        if mod(s, 2) == 0 || mod(s, 3) == 0 || mod(s, 5) == 0
            lengths[i] = s
            total_length *= s
        else
            throw(ArgumentError("""Plans can only have dims that are
                                   powers of 2, 3, or 5"""))
        end
    end
    if T <: clfftSingle
        if total_length > SP_MAX_LEN
            throw(ArgumentError("""CLFFT supports single precision transform
                                   lengths up to $(SP_MAX_LEN)"""))
        end
    else
        if total_length > DP_MAX_LEN
            throw(ArgumentError("""clFFT supports double precision transform
                                 lengths up to $(DP_MAX_LEN)"""))
        end
    end
    ph = PlanHandle[0]
    err = api.clfftCreateDefaultPlan(ph, ctx.id, Int32(ndim), lengths)
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
    plan = Plan{T}(ph)

    tdim = ndim
    tstrides = strides(input)
    # TODO : this works for dense arrays
    # need to test out for arbitrary regions
    tdistance = 0
    tbatchsize = 1

    set_layout!(plan, :interleaved, :interleaved)
    set_instride!(plan, tstrides)
    set_outstride!(plan, tstrides)
    set_distance!(plan, tdistance, tdistance)
    set_batchsize!(plan, tbatchsize)

    return plan
end

Plan{T<:clfftNumber}(::Type{T}, ctx::cl.Context, input::StridedArray{T}) =
        Plan(T, ctx, input, 1:ndims(input))

precision(p::Plan) = begin
    res = Cint[0]
    @check api.clfftGetPlanPrecision(p.id[1], res)
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


set_precision!(p::Plan, v::Symbol) = begin
    if v == :single
        @check api.clfftSetPlanPrecision(p.id[1], api.CLFFT_SINGLE)
    elseif v == :double
        @check api.clfftSetPlanPrecision(p.id[1], api.CLFFT_DOUBLE)
    elseif v == :single_fast
        @check api.clfftSetPlanPrecision(p.id[1], api.CLFFT_SINGLE_FAST)
    elseif v == :double_fast
        @check api.clfftSetPlanPrecision(p.id[1], api.CLFFT_DOUBLE_FAST)
    else
        error("unknown precision $v")
    end
    return p
end


layout(p::Plan) = begin
    i = Cint[0]
    o = Cint[0]
    @check api.clfftGetLayout(p.id[1], i, o)
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


set_layout!(p::Plan, in::Symbol, out::Symbol) = begin
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
    @check api.clfftSetLayout(p.id[1], args[1], args[2])
    return p
end


result(p::Plan) = begin
    res = Cint[0]
    @check api.clfftGetResultLocation(p.id[1], res)
    if res[1] == 1
        return :inplace
    elseif res[1] == 2
        return :outofplace
    else
        error("undefined")
    end
end


set_result!(p::Plan, v::Symbol) = begin
    if v == :inplace
        @check api.clfftSetResultLocation(p.id[1], api.CLFFT_INPLACE)
    elseif v == :outofplace
        @check api.clfftSetResultLocation(p.id[1], api.CLFFT_OUTOFPLACE)
    else
        throw(ArgumentError("set_result must be :inplace or :outofplace"))
    end
    return p
end


scaling_factor(p::Plan, dir::Symbol) = begin
    res = Cint[0]
    d::Cint
    if dir == :forward
        d = Int32(-1)
    elseif dir == :backward
        d = Int32(1)
    else
        error("undefined")
    end
    scale = Float32[0]
    @check api.clfftGetPlanScale(p.id[1], d, scale)
    return scale[1]
end


set_scaling_factor!(p::Plan, dir::Symbol, f::AbstractFloat) = begin
    if dir == :forward
        d = Int32(-1)
    elseif d == :backward
        d = Int32(1)
    else
        error("undefined")
    end
    @check api.clfftsetPlanScale(p.id[1], d, float32(f))
    return p
end


set_batchsize!(p::Plan, n::Integer) = begin
    @assert n > 0
    @check api.clfftSetPlanBatchSize(p.id[1], convert(Csize_t, n))
    return p
end


batchsize(p::Plan) = begin
    res = Csize_t[0]
    @check api.clfftGetPlanBatchSize(p.id[1], res)
    return Int(res[1])
end


set_dim!(p::Plan, d::Integer) = begin
    @assert d > 0 && d <= 3
    @check api.clfftSetPlanDim(p.id[1], convert(Csize_t, d))
    return p
end


dim(p::Plan) = begin
    res  = Int32[0]
    size = cl.CL_uint[0]
    @check api.clfftGetPlanDim(p.id[1], res, size)
    return Int(res[1])
end


set_lengths!(p::Plan, dims::Dims) = begin
    ndim = length(dims)
    @assert ndim <= 3
    nd = Array(Csize_t, ndim)
    for (i, d) in enumerate(dims)
        nd[i] = d
    end
    @check api.clfftSetPlanLength(p.id[1], Int32(ndim), nd)
    return p
end


lengths(p::Plan) = begin
    d = dim(p)
    res = Array(Csize_t, d)
    @check api.clfftGetPlanLength(p.id[1], Int32(d), res)
    return map(Int, res)
end


instride(p::Plan) = begin
    d = dim(p)
    res = Array(Csize_t, d)
    @check api.clfftGetPlanInStride(p.id[1], Int32(d), res)
    return map(Int, res)
end


set_instride!(p::Plan, instrides) = begin
    d = length(instrides)
    @assert d == dim(p)
    strides = Csize_t[Int(s) for s in instrides]
    @check api.clfftSetPlanInStride(p.id[1], Int32(d), strides)
    return p
end


outstride(p::Plan) = begin
    d = dim(p)
    res = Array(Csize_t, d)
    @check api.clfftGetPlanOutStride(p.id[1], Int32(d), res)
    return map(Int, res)
end


set_outstride!(p::Plan, outstrides) = begin
    d = length(outstrides)
    @assert d == dim(p)
    strides = Csize_t[Int(s) for s in outstrides]
    @check api.clfftSetPlanInStride(p.id[1], Int32(d), strides)
    return p
end


distance(p::Plan) = begin
    indist = Csize_t[0]
    odist  = Csize_t[0]
    @check api.clfftGetPlanDistance(p.id[1], indist, odist)
    return (indist[1], odist[1])
end


set_distance!(p::Plan, indist::Integer, odist::Integer) = begin
    @assert indist >= 0 && odist >= 0
    i = UInt(indist)
    o = UInt(odist)
    @check api.clfftSetPlanDistance(p.id[1], i, o)
    return p
end


transpose_result(p::Plan) = begin
    res = Cint[0]
    @check api.clfftGetPlanTransposeResult(p.id[1], res)
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
        @check api.clfftSetTransposeResult(p.id[1], d,
                cl.CLFFT_TRANSPOSED)
    else
        @check api.clfftSetTransposeResult(p.id[1], d,
                cl.CLFFT_NOTRANSPOSE)
    end
end


tmp_buffer_size(p::Plan) = begin
    res = Csize_t[0]
    @check api.clfftGetTmpBufSize(p.id[1], res)
    return Int(res[1])
end


context(p::Plan) = begin
    res = Array(cl.CL_context)
    @check api.clfftGetPlanContext(p.id[1], res)
    return cl.Context(res[1])
end


bake!(p::Plan, qs::Vector{cl.CmdQueue}) = begin
    nqueues = length(qs)
    q_ids = [q.id for q in qs]
    @check api.clfftBakePlan(p.id[1], nqueues, q_ids, C_NULL, C_NULL)
    return p
end
bake!(p::Plan, q::cl.CmdQueue) = bake!(p, [q])


function enqueue_transform{T<:clfftNumber}(p::Plan,
                                           dir::Symbol,
                                           qs::Vector{cl.CmdQueue},
                                           input::cl.Buffer{T},
                                           output::Union{Void,cl.Buffer{T}};
                                           wait_for::Union{Void,Vector{cl.Event}}=nothing,
                                           tmp::Union{Void,cl.Buffer{T}}=nothing)
    if dir != :forward && dir != :backward
        throw(ArgumentError("Unknown direction $dir"))
    end
    q_ids = [q.id for q in qs]
    in_buff_ids  = [input.id]
    out_buff_ids = C_NULL
    if output != nothing
        out_buff_ids = [output.id]
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
                              p.id[1],
                              dir == :forward ? api.CLFFT_FORWARD : api.CLFFT_BACKWARD,
                              UInt32(nqueues),
                              q_ids,
                              UInt32(nevts),
                              evt_ids,
                              out_evts,
                              in_buff_ids,
                              out_buff_ids,
                              tmp_buffer)
    return [cl.Event(e_id) for e_id in out_evts]
end

end # module
