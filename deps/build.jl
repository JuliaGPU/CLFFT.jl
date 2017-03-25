using BinDeps
@BinDeps.setup
libnames = ["libCLFFT", "clFFT", "libclFFT"]
libCLFFT = library_dependency("libCLFFT", aliases = libnames)
archive = "package"
libpath = "package/bin"
baseurl = "https://github.com/clMathLibraries/clFFT/releases/download/v2.12.2/clFFT-2.12.2-"
# download a pre-compiled binary (built by GLFW)
if is_windows()
    if Sys.ARCH == :x86_64
        uri = URI(baseurl * "Windows-x64.zip")
        provides(
            Binaries, uri,
            libCLFFT, unpacked_dir = archive,
            installed_libpath = libpath, os = :Windows
        )
    else
        error("Only 64 bits windows supported with automatic build")
    end
end

if is_linux()
    provides(AptGet, "libclfft-dev", libCLFFT)
    if Sys.ARCH == :x86_64
        uri = URI(baseurl * "Linux-x64.tar.gz")
        provides(
            Binaries, uri,
            libCLFFT, unpacked_dir = archive,
            installed_libpath = libpath, os = :Linux
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
