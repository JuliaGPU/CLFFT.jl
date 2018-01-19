using BinDeps
using Compat

@BinDeps.setup
libnames = ["libCLFFT", "clFFT", "libclFFT"]
libCLFFT = library_dependency("libCLFFT", aliases = libnames)
version = "2.12.2"
baseurl = "https://github.com/clMathLibraries/clFFT/releases/download/v$(version)/clFFT-$(version)-"

# download a pre-compiled binary (built by GLFW)
if is_windows()
    if Sys.ARCH == :x86_64
        uri = URI(baseurl * "Windows-x64.zip")
        basedir = joinpath(@__DIR__, "clFFT-$(version)-Windows-x64")
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
        basedir = joinpath(@__DIR__, "clFFT-$(version)-Linux-x64")
        provides(
            Binaries, uri,
            libCLFFT, unpacked_dir = basedir,
            installed_libpath = joinpath(basedir, "lib64"), os = :Linux
        )
    end
end

if is_apple()
    using Homebrew
    provides(Homebrew.HB, "homebrew/core/clfft", libCLFFT, os = :Darwin)
end

@BinDeps.install Dict("libCLFFT" => "libCLFFT")
