using BinDeps
using Compat

@BinDeps.setup
libnames = ["libCLFFT", "clFFT", "libclFFT"]
libCLFFT = library_dependency("libCLFFT", aliases = libnames)
baseurl = "https://github.com/clMathLibraries/clFFT/releases/download/v2.12.2/clFFT-2.12.2-"

# download a pre-compiled binary (built by GLFW)
if is_windows()
    if Sys.ARCH == :x86_64
        uri = URI(baseurl * "Windows-x64.zip")
        basedir = joinpath(@__DIR__, "clFFT-2.12.2-Windows-x64")
        provides(
            Binaries, uri,
            libCLFFT, unpacked_dir = basedir,
            installed_libpath = joinpath(basedir, "bin"), os = :Windows
        )
    else
        error("Only 64 bits windows supported with automatic build")
    end
end

if is_linux()
    provides(AptGet, "libclfft-dev", libCLFFT)
    if Sys.ARCH == :x86_64
        uri = URI(baseurl * "Linux-x64.tar.gz")
        basedir = joinpath(@__DIR__, "clFFT-2.12.2-Linux-x64")
        provides(
            Binaries, uri,
            libCLFFT, unpacked_dir = basedir,
            installed_libpath = joinpath(basedir, "bin"), os = :Linux
        )
    end
end
if is_apple()
    error("""
        OSX not oficially supported.
        Find manual build instructions on: https://github.com/clMathLibraries/clBLAS/wiki/Build
    """)
end

@BinDeps.install Dict("libCLFFT" => "libCLFFT")
