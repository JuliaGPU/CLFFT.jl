
using Libdl

libnames = ["libCLFFT", "clFFT", "libclFFT"]

for l in libnames
    global libname = Libdl.find_library(l)
    if (libname != "")
        break
    end
end

if (libname == "")
    error("CLFFT library not installed.")
else
    global libCLFFT = Libdl.dlpath(libname)
end

