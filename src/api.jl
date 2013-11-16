module api

import OpenCL
const cl = OpenCL

const libopencl = "clFFT"

macro clfft(func, arg_types)
    local args_in  = Symbol[symbol("arg$i::$T")
                            for (i, T) in enumerate(arg_types.args)]
    local funcname = symbol("clfft$func")
    quote
        $(esc(funcname))($(args_in...)) = ccall(($(string(funcname)), libclfft),
                                                 cl.CL_int, #clfftStatus
                                                 $arg_types,
                                                 $(args_in...))
    end
end

immutable clfftSetupData
    major::cl.CL_uint
    minor::cl.CL_uint
    patch::cl.CL_path
    debug_flags::cl.CL_ulong
end

typealias PlanHandle Csize_t
typealias Callback   Any 
typealias UserData   Ptr{Void}

# @brief Initialize internal FFT resources.
# @details AMD's FFT implementation caches kernels, programs and buffers for its internal use.
# @param[in] setupData Data structure that can be passed into the setup routine to control FFT generation behavior
#            and debug functionality
# @return    Enum describing error condition; superset of OpenCL error codes
@clfft(Setup, (Ptr{clfftSetupData},)

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
@clfft(CreateDefaultPlan, (Ptr{PlanHandle}, cl.CL_context, clfftDim, Ptr{Csize_t}))

# @brief Create a copy of an existing plan.
# @details This API allows a client to create a new plan based upon an existing plan.  This is a convenience function
# provided for quickly creating plans that are similar, but may differ slightly.
# @param[out] out_plHandle Handle to the newly created plan that is based on in_plHandle
# @param[in] new_context Client is responsible for providing a new context for the new plan
# @param[in] in_plHandle Handle to a plan to be copied, previously created
# @return Enum describing error condition; superset of OpenCL error codes
@clfft(CopyPlan, (Ptr{PlanHandle}, cl.CL_context, clfftDim, Ptr{Csize_t}))

# @brief Prepare the plan for execution.
# @details After all plan parameters are set, the client has the option of 'baking' the plan, which tells the runtime that
# no more changes to the plan's parameters are expected, and the OpenCL kernels should be compiled.  This optional function
# allows the client application to perform this function when the application is being initialized 
# instead of on the first execution.
# At this point, the clfft runtime will apply all implimented optimizations, possibly including
# running kernel experiments on the devices in the plan context.
# Users should assume that this function will take a long time to execute.  If a plan is not baked before being executed,
# users should assume that the first call to clfftEnqueueTransform will take a long time to execute.
# If any significant parameter of a plan is changed after the plan is baked (by a subsequent call to one of
# the clfftSetPlan____ functions), that will not be considered an error.  Instead, the plan will revert back to
# the unbaked state, discarding the benefits of the baking operation.

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
@clfft(BakePlan, (PlanHandle, cl.CL_uint, Ptr{cl.CL_command_queue}, CallBack, UserData))
