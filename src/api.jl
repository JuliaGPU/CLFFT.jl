module api

import OpenCL.cl

@static if is_unix() const libclfft = "libclFFT" end
@static if is_windows() const libclfft = "clFFT" end


macro clfft(func, arg_types)
    local args_in  = Symbol[Symbol("arg$i::$T")
                            for (i, T) in enumerate(arg_types.args)]
    local funcname = Symbol("clfft$func")
    @eval begin
        $(funcname)($(args_in...)) = ccall(($(string(funcname)), libclfft),
                                                 cl.CL_int, #clfftStatus
                                                 $arg_types,
                                                 $(args_in...))
    end
end

type SetupData
    major::cl.CL_uint
    minor::cl.CL_uint
    patch::cl.CL_uint
    debug_flags::cl.CL_ulong

    function SetupData(major::Integer,
                       minor::Integer,
                       patch::Integer,
                       debug_flags::Integer)
        d = new(cl.cl_uint(major),
                cl.cl_uint(minor),
                cl.cl_uint(patch),
                cl.cl_ulong(debug_flags))
        setup = [d]
        error = clfftSetup(setup)
        if error != CLFFT_SUCCESS
            error("Failed to setup CLFFT Library")
        end
        finalizer(d, x -> begin
            # gc'ing before teardown
            # helps prevent a double-free event for Plan objects
            # A kludge to enforce gc order.
            Base.gc()
            clfftTeardown()
        end)
        return d
    end
end


typealias PlanHandle Csize_t
typealias Callback   Ptr{Void}
typealias UserData   Ptr{Void}

typealias Precision  Cint
typealias Dim        Cint
typealias Direction  Cint
typealias Layout     Cint
typealias ResultLocation   Cint
typealias ResultTransposed Cint

clfft_dim(x) = convert(Dim, x)
clfft_direction(x) = convert(Direction, x)

# ERROR CODES
const CLFFT_INVALID_GLOBAL_WORK_SIZE          = cl.CL_INVALID_GLOBAL_WORK_SIZE
const CLFFT_INVALID_MIP_LEVEL                 = cl.CL_INVALID_MIP_LEVEL
const CLFFT_INVALID_BUFFER_SIZE               = cl.CL_INVALID_BUFFER_SIZE
const CLFFT_INVALID_GL_OBJECT                 = cl.CL_INVALID_GL_OBJECT
const CLFFT_INVALID_OPERATION                 = cl.CL_INVALID_OPERATION
const CLFFT_INVALID_EVENT                     = cl.CL_INVALID_EVENT
const CLFFT_INVALID_EVENT_WAIT_LIST           = cl.CL_INVALID_EVENT_WAIT_LIST,
const CLFFT_INVALID_GLOBAL_OFFSET             = cl.CL_INVALID_GLOBAL_OFFSET,
const CLFFT_INVALID_WORK_ITEM_SIZE            = cl.CL_INVALID_WORK_ITEM_SIZE
const CLFFT_INVALID_WORK_GROUP_SIZE           = cl.CL_INVALID_WORK_GROUP_SIZE
const CLFFT_INVALID_WORK_DIMENSION            = cl.CL_INVALID_WORK_DIMENSION
const CLFFT_INVALID_KERNEL_ARGS               = cl.CL_INVALID_KERNEL_ARGS
const CLFFT_INVALID_ARG_SIZE                  = cl.CL_INVALID_ARG_SIZE
const CLFFT_INVALID_ARG_VALUE                 = cl.CL_INVALID_ARG_VALUE
const CLFFT_INVALID_ARG_INDEX                 = cl.CL_INVALID_ARG_INDEX
const CLFFT_INVALID_KERNEL                    = cl.CL_INVALID_KERNEL
const CLFFT_INVALID_KERNEL_DEFINITION         = cl.CL_INVALID_KERNEL_DEFINITION
const CLFFT_INVALID_KERNEL_NAME               = cl.CL_INVALID_KERNEL_NAME
const CLFFT_INVALID_PROGRAM_EXECUTABLE        = cl.CL_INVALID_PROGRAM_EXECUTABLE
const CLFFT_INVALID_PROGRAM                   = cl.CL_INVALID_PROGRAM
const CLFFT_INVALID_BUILD_OPTIONS             = cl.CL_INVALID_BUILD_OPTIONS
const CLFFT_INVALID_BINARY                    = cl.CL_INVALID_BINARY
const CLFFT_INVALID_SAMPLER                   = cl.CL_INVALID_SAMPLER
const CLFFT_INVALID_IMAGE_SIZE                = cl.CL_INVALID_IMAGE_SIZE
const CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR   = cl.CL_INVALID_IMAGE_FORMAT_DESCRIPTOR
const CLFFT_INVALID_MEM_OBJECT                = cl.CL_INVALID_MEM_OBJECT
const CLFFT_INVALID_HOST_PTR                  = cl.CL_INVALID_HOST_PTR
const CLFFT_INVALID_COMMAND_QUEUE             = cl.CL_INVALID_COMMAND_QUEUE
const CLFFT_INVALID_QUEUE_PROPERTIES          = cl.CL_INVALID_QUEUE_PROPERTIES
const CLFFT_INVALID_CONTEXT                   = cl.CL_INVALID_CONTEXT
const CLFFT_INVALID_DEVICE                    = cl.CL_INVALID_DEVICE
const CLFFT_INVALID_PLATFORM                  = cl.CL_INVALID_PLATFORM
const CLFFT_INVALID_DEVICE_TYPE               = cl.CL_INVALID_DEVICE_TYPE
const CLFFT_INVALID_VALUE                     = cl.CL_INVALID_VALUE
const CLFFT_MAP_FAILURE                       = cl.CL_MAP_FAILURE
const CLFFT_BUILD_PROGRAM_FAILURE             = cl.CL_BUILD_PROGRAM_FAILURE
const CLFFT_IMAGE_FORMAT_NOT_SUPPORTED        = cl.CL_IMAGE_FORMAT_NOT_SUPPORTED
const CLFFT_IMAGE_FORMAT_MISMATCH             = cl.CL_IMAGE_FORMAT_MISMATCH
const CLFFT_MEM_COPY_OVERLAP                  = cl.CL_MEM_COPY_OVERLAP
const CLFFT_PROFILING_INFO_NOT_AVAILABLE      = cl.CL_PROFILING_INFO_NOT_AVAILABLE
const CLFFT_OUT_OF_HOST_MEMORY                = cl.CL_OUT_OF_HOST_MEMORY
const CLFFT_OUT_OF_RESOURCES                  = cl.CL_OUT_OF_RESOURCES
const CLFFT_MEM_OBJECT_ALLOCATION_FAILURE     = cl.CL_MEM_OBJECT_ALLOCATION_FAILURE
const CLFFT_COMPILER_NOT_AVAILABLE            = cl.CL_COMPILER_NOT_AVAILABLE
const CLFFT_DEVICE_NOT_AVAILABLE              = cl.CL_DEVICE_NOT_AVAILABLE
const CLFFT_DEVICE_NOT_FOUND                  = cl.CL_DEVICE_NOT_FOUND
const CLFFT_SUCCESS                           = cl.CL_SUCCESS

const CLFFT_BUGCHECK                  = Int32(4*1024)
const CLFFT_NOTIMPLEMENTED            = Int32(4*1024+1)  # Functionality is not implemented yet.
const CLFFT_TRANSPOSED_NOTIMPLEMENTED = Int32(4*1024+2)  # Transposed functionality is not implemented for this transformation.
const CLFFT_FILE_NOT_FOUND            = Int32(4*1024+3)  # Tried to open an existing file on the host system, but failed.
const CLFFT_FILE_CREATE_FAILURE       = Int32(4*1024+4)  # Tried to create a file on the host system, but failed.
const CLFFT_VERSION_MISMATCH          = Int32(4*1024+5)  # Version conflict between client and library.
const CLFFT_INVALID_PLAN              = Int32(4*1024+6)  # Requested plan could not be found.
const CLFFT_DEVICE_NO_DOUBLE          = Int32(4*1024+7)  # Double precision not supported on this device.

const CLFFT_1D = Int32(1) # 1 Dimensional FFT transform (default).
const CLFFT_2D = Int32(2) # 2 Dimensional FFT transform.
const CLFFT_3D = Int32(3) # 3 Dimensional FFT transform.

const CLFFT_COMPLEX_INTERLEAVED   = Int32(1) # An array of complex numbers, with real and imaginary components together (default).
const CLFFT_COMPLEX_PLANAR        = Int32(2) # Arrays of real componets and arrays of imaginary components that have been seperated out.
const CLFFT_HERMITIAN_INTERLEAVED = Int32(3) # Compressed form of complex numbers; complex-conjugates not stored, real and imaginary components in same array.
const CLFFT_HERMITIAN_PLANAR      = Int32(4) # Compressed form of complex numbers; complex-conjugates not stored, real and imaginary components in separate arrays.
const CLFFT_REAL                  = Int32(5) # An array of real numbers, with no corresponding imaginary components.

const CLFFT_SINGLE       = Int32(1) # An array of complex numbers, with real and imaginary components as floats (default).
const CLFFT_DOUBLE       = Int32(2) # An array of complex numbers, with real and imaginary components as doubles.
const CLFFT_SINGLE_FAST  = Int32(3) # Faster implementation preferred.
const CLFFT_DOUBLE_FAST  = Int32(4) # Faster implementation preferred.

const CLFFT_FORWARD      = Int32(-1) # FFT transform from the time to the frequency domain.
const CLFFT_BACKWARD     = Int32(1)  # FFT transform from the frequency to the time domain
const CLFFT_MINUS        = Int32(-1) # Alias for the forward transform.
const CLFFT_PLUS         = Int32(1)  # Alias for the backward transform.

const CLFFT_INPLACE      = Int32(1) # The input and output buffers are the same (default).
const CLFFT_OUTOFPLACE   = Int32(2) # Seperate input and output buffers.

const CLFFT_NOTRANSPOSE   = Int32(1) # The results are returned in the original preserved order (default)
const CLFFT_TRANSPOSED    = Int32(2) # The result is transposed where transpose kernel is supported (possibly faster)

# @brief Initialize an clfftSetupData struct for the client
# @details clfftSetupData is passed to clfftSetup to control behavior of the FFT runtime
# @param[out] setupData Data structure is cleared,
#             initialized with version information and default values
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(InitSetupData, (Ptr{SetupData},))

# @brief Initialize internal FFT resources.
# @details AMD's FFT implementation caches kernels, programs and buffers for its internal use.
# @param[in] setupData Data structure that can be passed into the setup routine to control FFT generation behavior
#            and debug functionality
# @return    Enum describing error condition; superset of OpenCL error codes
@clfft(Setup, (Ptr{SetupData},))

# @brief Release all internal resources.
# @details Call when client is done with this FFT library, allowing the library to destroy all resources it has cached
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(Teardown, ())

# @brief Query the FFT library for version information
# @details Return the major, minor and patch version numbers associated with this FFT library
# @param[out] major Major functionality change
# @param[out] minor Minor functionality change
# @param[out] patch Bug fixes, documentation changes, no new features introduced
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(GetVersion, (Ptr{cl.CL_uint}, Ptr{cl.CL_uint}, Ptr{cl.CL_uint}))

# @brief Create a plan object initialized entirely with default values.
# @details A plan is a repository of state for calculating FFT's.  Allows the runtime to pre-calculate kernels, programs
# and buffers and associate them with buffers of specified dimensions.
# @param[out] plHandle Handle to the newly created plan
# @param[in] context Client is responsible for providing an OpenCL context for the plan
# @param[in] dim The dimensionality of the FFT transform; describes how many elements are in the array
# @param[in] clLengths An array of lengths, of size 'dim'.  Each value describes the length of additional dimensions
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(CreateDefaultPlan, (Ptr{PlanHandle}, cl.CL_context, Dim, Ptr{Csize_t}))

# @brief Create a copy of an existing plan.
# @details This API allows a client to create a new plan based upon an existing plan.  This is a convenience function
# provided for quickly creating plans that are similar, but may differ slightly.
# @param[out] out_plHandle Handle to the newly created plan that is based on in_plHandle
# @param[in] new_context Client is responsible for providing a new context for the new plan
# @param[in] in_plHandle Handle to a plan to be copied, previously created
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(CopyPlan, (Ptr{PlanHandle}, cl.CL_context, Dim, Ptr{Csize_t}))

# @brief Prepare the plan for execution.
# @details After all plan parameters are set, the client has the option of 'baking' the plan, which tells the runtime that
# no more changes to the plan's parameters are expected, and the OpenCL kernels should be compiled.
# This optional function allows the client application to perform this function when the
# application is being initialized instead of on the first execution.
# At this point, the clfft runtime will apply all implimented optimizations,
# possibly including running kernel experiments on the devices in the plan context.
# Users should assume that this function will take a long time to execute.
# If a plan is not baked before being executed, users should assume that the first call to clfftEnqueueTransform will take a long time to execute.
# If any significant parameter of a plan is changed after the plan is baked (by a subsequent call to one of
# the clfftSetPlan____ functions), that will not be considered an error.
# Instead, the plan will revert back to the unbaked state, discarding the benefits of the baking operation.

# @param[in] plHandle Handle to a plan previously created
# @param[in] numQueues Number of command queues in commQueueFFT; 0 is a valid value, in which case client does not want
#            the runtime to run load experiments and only pre-calculate state information
# @param[in] commQueueFFT An array of cl_command_queues created by the client; the command queues must be a proper subset of
#            the devices included in the plan context
# @param[in] pfn_notify A function pointer to a notification routine. The notification routine is a callback function that
#            an application can register and which will be called when the program executable has been built (successfully or unsuccessfully).
#            Currently, this parameter MUST be NULL or nullptr.
# @param[in] user_data Passed as an argument when pfn_notify is called.
#            Currently, this parameter MUST be NULL or nullptr.
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(BakePlan, (PlanHandle, cl.CL_uint, Ptr{cl.CL_command_queue}, Callback, UserData))

# @brief Release the resources of a plan.
# @details A plan may include kernels, programs and buffers associated with it that consume memory. When a plan  is not needed anymore, the client should release the plan.
# @param[in,out] plHandle Handle to a plan previously created
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(DestroyPlan, (Ptr{PlanHandle},))

# @brief Retrieve the OpenCL context of a previously created plan.
# @details User should pass a reference to an cl_context variable, which will be changed to point to a
# context set in the specified plan.
# @param[in] plHandle Handle to a plan previously created
# @param[out] context Reference to user allocated cl_context, which will point to context set in plan
 # @return Enum describing error condition; superset of OpenCL error codes
@clfft(GetPlanContext, (PlanHandle, Ptr{cl.CL_context}))

# @brief Retrieve the floating point precision of the FFT data
# @details User should pass a reference to an clfftPrecision variable, which will be set to the
# precision of the FFT complex data in the plan.
# @param[in] plHandle Handle to a plan previously created
# @param[out] precision Reference to user clfftPrecision enum
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(GetPlanPrecision, (PlanHandle, Ptr{Precision}))

# @brief Set the floating point precision of the FFT data
# @details Set the plan property which will be the precision of the FFT complex data in the plan.
# @param[in] plHandle Handle to a plan previously created
# @param[in] precision Reference to user clfftPrecision enum
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(SetPlanPrecision, (PlanHandle, Precision))

# @brief Retrieve the scaling factor that should be applied to the FFT data
# @details User should pass a reference to an cl_float variable, which will be set to the
# floating point scaling factor that will be multiplied across the FFT data.
# @param[in] plHandle Handle to a plan previously created
# @param[in] dir Which direction does the scaling factor apply to
# @param[out] scale Reference to user cl_float variable
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(GetPlanScale, (PlanHandle, Direction, Ptr{cl.CL_float}))

# @brief Set the scaling factor that should be applied to the FFT data
# @details Set the plan property which will be the floating point scaling factor that will be
# multiplied across the FFT data.
# @param[in] plHandle Handle to a plan previously created
# @param[in] dir Which direction does the scaling factor apply to
# @param[in] scale Reference to user cl_float variable
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(SetPlanScale, (PlanHandle, Direction, cl.CL_float))

# @brief Retrieve the number of discrete arrays that this plan can handle concurrently
# @details User should pass a reference to an cl_uint variable, which will be set to the
# number of discrete arrays (1D or 2D) that will be batched together for this plan
# @param[in] plHandle Handle to a plan previously created
# @param[out] batchSize How many discrete number of FFT's are to be performed
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(GetPlanBatchSize, (PlanHandle, Ptr{Csize_t}))

# @brief Set the number of discrete arrays that this plan can handle concurrently
# @details Set the plan property which will be set to the number of discrete arrays (1D or 2D)
# that will be batched together for this plan
# @param[in] plHandle Handle to a plan previously created
# @param[in] batchSize How many discrete number of FFT's are to be performed
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(SetPlanBatchSize, (PlanHandle, Csize_t))

# @brief Retrieve the dimensionality of FFT's to be transformed in the plan
# @details Queries a plan object and retrieves the dimensionality that the plan is set for.  A size is returned to
# help the client allocate the proper storage to hold the dimensions in a further call to clfftGetPlanLength
# @param[in] plHandle Handle to a plan previously created
# @param[out] dim The dimensionality of the FFT's to be transformed
# @param[out] size Value used to allocate an array to hold the FFT dimensions.
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(GetPlanDim, (PlanHandle, Ptr{Dim}, Ptr{cl.CL_uint}))

# @brief Set the dimensionality of FFT's to be transformed by the plan
# @details Set the dimensionality of FFT's to be transformed by the plan
# @param[in] plHandle Handle to a plan previously created
# @param[in] dim The dimensionality of the FFT's to be transformed
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(SetPlanDim, (PlanHandle, Dim))

# @brief Retrieve the length of each dimension of the FFT
# @details User should pass a reference to a size_t array, which will be set to the
# length of each discrete dimension of the FFT
# @param[in] plHandle Handle to a plan previously created
# @param[in] dim The dimension of the length parameters; describes how many elements are in the array
# @param[out] clLengths An array of lengths, of size 'dim'.  Each array value describes the length of each dimension
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(GetPlanLength, (PlanHandle, Dim, Ptr{Csize_t}))

# @brief Set the length of each dimension of the FFT
# @details Set the plan property which will be the length of each discrete dimension of the FFT
# @param[in] plHandle Handle to a plan previously created
# @param[in] dim The dimension of the length parameters; describes how many elements are in the array
# @param[in] clLengths An array of lengths, of size 'dim'.  Each value describes the length of additional dimensions
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(SetPlanLength, (PlanHandle, Dim, Ptr{Csize_t}))

# @brief Retrieve the distance between consecutive elements for input buffers in a dimension.
# @details Depending on how the dimension is set in the plan (for 2D or 3D FFT's), strideY or strideZ can be safely ignored
# @param[in] plHandle Handle to a plan previously created
# @param[in] dim The dimension of the stride parameters; describes how many elements are in the array
# @param[out] clStrides An array of strides, of size 'dim'.
@clfft(GetPlanInStride, (PlanHandle, Dim, Ptr{Csize_t}))

# @brief Set the distance between consecutive elements for input buffers in a dimension.
# @details Set the plan properties which will be the distance between elements in a given dimension (units are in terms of clfftPrecision)
# @param[in] plHandle Handle to a plan previously created
# @param[in] dim The dimension of the stride parameters; describes how many elements are in the array
# @param[in] clStrides An array of strides, of size 'dim'. Usually strideX=1 so that successive elements in the first dimension are stored contiguously.
# Typically strideY=LenX, strideZ=LenX*LenY such that successive elements in the second and third dimensions are stored in packed format.
@clfft(SetPlanInStride, (PlanHandle, Dim, Ptr{Csize_t}))

# @brief Retrieve the distance between consecutive elements for output buffers in a dimension.
# @details Depending on how the dimension is set in the plan (for 2D or 3D FFT's), strideY or strideZ can be safely ignored
# @param[in] plHandle Handle to a plan previously created
# @param[in] dim The dimension of the stride parameters; describes how many elements are in the array
# @param[out] clStrides An array of strides, of size 'dim'.
@clfft(GetPlanOutStride, (PlanHandle, Dim, Ptr{Csize_t}))

# @brief Set the distance between consecutive elements for output buffers in a dimension.
# @details Set the plan properties which will be the distance between elements in a given dimension
# (units are in terms of clfftPrecision)
# @param[in] plHandle Handle to a plan previously created
# @param[in] dim The dimension of the stride parameters; describes how many elements are in the array
# @param[in] clStrides An array of strides, of size 'dim'.  Usually strideX=1 so that successive elements in the first dimension are stored contiguously.
# Typically strideY=LenX, strideZ=LenX*LenY such that successive elements in the second and third dimensions are stored in packed format.
@clfft(SetPlanOutStride, (PlanHandle, Dim, Ptr{Csize_t}))

# @brief Retrieve the distance between Array objects
# @details Pitch is the distance between each discrete array object in an FFT array. This is only used
# for 'array' dimensions in clfftDim; see clfftSetPlanDimension (units are in terms of clfftPrecision)
# @param[in]  plHandle Handle to a plan previously created
# @param[out] iDist The distance between the beginning elements of the discrete array objects in memory on input.
#             For contiguous arrays in memory, iDist=(strideX*strideY*strideZ)
# @param[out] oDist The distance between the beginning elements of the discrete array objects in memory on output.
#             For contiguous arrays in memory, oDist=(strideX*strideY*strideZ)
@clfft(GetPlanDistance, (PlanHandle, Ptr{Csize_t}, Ptr{Csize_t}))

# @brief Set the distance between Array objects
# @details Pitch is the distance between each discrete array object in an FFT array. This is only used
# for 'array' dimensions in clfftDim; see clfftSetPlanDimension (units are in terms of clfftPrecision)
# @param[in] plHandle Handle to a plan previously created
# @param[out] iDist The distance between the beginning elements of the discrete array objects in memory on input.
#             For contiguous arrays in memory, iDist=(strideX*strideY*strideZ)
# @param[out] oDist The distance between the beginning elements of the discrete array objects in memory on output.
#             For contiguous arrays in memory, oDist=(strideX*strideY*strideZ)
@clfft(SetPlanDistance, (PlanHandle, Csize_t, Csize_t))

# @brief Retrieve the expected layout of the input and output buffers
# @details Output buffers can be filled with either hermitian or complex numbers.  Complex numbers can be stored
# in various layouts; this informs the FFT engine what layout to produce on output
# @param[in] plHandle Handle to a plan previously created
# @param[out] iLayout Indicates how the input buffers are laid out in memory
# @param[out] oLayout Indicates how the output buffers are laid out in memory
@clfft(GetLayout, (PlanHandle, Ptr{Layout}, Ptr{Layout}))

# @brief Set the expected layout of the input and output buffers
# @details Output buffers can be filled with either hermitian or complex numbers.  Complex numbers can be stored
# in various layouts; this informs the FFT engine what layout to produce on output
# @param[in] plHandle Handle to a plan previously created
# @param[in] iLayout Indicates how the input buffers are laid out in memory
# @param[in] oLayout Indicates how the output buffers are laid out in memory
@clfft(SetLayout, (PlanHandle, Layout, Layout))

# @brief Retrieve whether the input buffers are going to be overwritten with results
# @details If the setting is to do an in-place transform, the input buffers are overwritten with the results of the
# transform.  If the setting is for out-of-place transforms, the engine knows to look for separate output buffers on the Enqueue call.
# @param[in] plHandle Handle to a plan previously created
# @param[out] placeness Tells the FFT engine to clobber the input buffers or to expect output buffers for results
@clfft(GetResultLocation, (PlanHandle, Ptr{ResultLocation}))

# @brief Set whether the input buffers are going to be overwritten with results
# @details If the setting is to do an in-place transform, the input buffers are overwritten with the results of the
# transform.  If the setting is for out-of-place transforms, the engine knows to look for separate output buffers on the Enqueue call.
# @param[in] plHandle Handle to a plan previously created
# @param[in] placeness Tells the FFT engine to clobber the input buffers or to expect output buffers for results
@clfft(SetResultLocation, (PlanHandle, ResultLocation))

# @brief Retrieve the final transpose setting of a muti-dimensional FFT
# @details A multi-dimensional FFT typically transposes the data several times during calculation.  If the client
# does not care about the final transpose to put data back in proper dimension, the final transpose can be skipped
# for possible speed improvements
# @param[in] plHandle Handle to a plan previously created
# @param[out] transposed Parameter specifies whether the final transpose can be skipped
@clfft(GetPlanTransposeResult, (PlanHandle, Ptr{ResultTransposed}))

# @brief Set the final transpose setting of a muti-dimensional FFT
# @details A multi-dimensional FFT typically transposes the data several times during calculation.  If the client
# does not care about the final transpose to put data back in proper dimension, the final transpose can be skipped
# for possible speed improvements
# @param[in] plHandle Handle to a plan previously created
# @param[in] transposed Parameter specifies whether the final transpose can be skipped
@clfft(SetPlanTransposeResult, (PlanHandle, ResultTransposed))

# @brief Get buffer size (in bytes), which may be needed internally for an intermediate buffer
# @details Very large FFT transforms may need multiple passes, and the operation would need a temporary buffer to hold
# intermediate results. This function is only valid after the plan is baked, otherwise an invalid operation error
# is returned. If buffersize returns as 0, the runtime needs no temporary buffer.
# @param[in]  plHandle Handle to a plan previously created
# @param[out] buffersize Size in bytes for intermediate buffer
@clfft(GetTmpBufSize, (PlanHandle, Ptr{Csize_t}))

# @brief Enqueue an FFT transform operation, and return immediately (non-blocking)
# @details This transform API is the function that actually computes the FFT transfrom. It is non-blocking as it
#          only enqueues the OpenCL kernels for execution. The synchronization step has to be managed by the user.
# @param[in]  plHandle Handle to a plan previously created
# @param[in]  dir Forwards or backwards transform
# @param[in]  numQueuesAndEvents Number of command queues in commQueues; number of expected events to be returned in outEvents
# @param[in]  commQueues An array of cl_command_queues created by the client; the command queues must be a proper subset of
#             the devices included in the plan context
# @param[in]  numWaitEvents Specify the number of elements in the eventWaitList array
# @param[in]  waitEvents Events that this transform should wait to complete before executing on the device
# @param[out] outEvents The runtime fills this array with events corresponding 1 to 1 with the input command queues passed
#             in commQueues.  This parameter can be NULL or nullptr, in which case client is not interested in receiving notifications
#             when transforms are finished, otherwise if not NULL the client is responsible for allocating this array, with at least
#             as many elements as specified in numQueuesAndEvents.
# @param[in]  inputBuffers An array of cl_mem objects that contain data for processing by the FFT runtime.  If the transform
#             is in place, the FFT results will overwrite the input buffers
# @param[out] outputBuffers An array of cl_mem objects that will store the results of out of place transforms.  If the transform
#             is in place, this parameter may be NULL or nullptr.  It is completely ignored
# @param[in]  tmpBuffer A cl_mem object that is reserved as a temporary buffer for FFT processing. If clTmpBuffers is NULL or nullptr,
#             and the runtime needs temporary storage, an internal temporary buffer will be created on the fly managed by the runtime.
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(EnqueueTransform, (PlanHandle,
                          Direction,
                          cl.CL_uint,
                          Ptr{cl.CL_command_queue},
                          cl.CL_uint,
                          Ptr{cl.CL_event},
                          Ptr{cl.CL_event},
                          Ptr{cl.CL_mem},
                          Ptr{cl.CL_mem},
                          cl.CL_mem))
end # end module
