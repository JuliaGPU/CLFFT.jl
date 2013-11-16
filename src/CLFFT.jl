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

end # module
