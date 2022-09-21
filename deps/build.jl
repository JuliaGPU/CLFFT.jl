using BinaryProvider
using InfoZIP

if Sys.ARCH != :x86_64
    error("Only 64 bits operational systems are supported with automatic build")
end

if Sys.islinux()
    so_name = "Linux"
elseif Sys.iswindows()
    so_name = "Windows"
else
    error("Only Linux or Windows are supported with automatic build")
end

# Download and install binaries
version = "2.12.2"
base_url = "https://github.com/clMathLibraries/clFFT/releases/download/v$(version)/clFFT-$(version)"
tarball_dir = joinpath(@__DIR__, "downloads/")

if Sys.islinux()
    tarball_url = "$base_url-Linux-x64.tar.gz"
    integrity_hash = "20c853aba91e725b2b946ea59d5e45791c163b096951e0812a5d1d72d9d6a7cb"
    
    download_verify_unpack(tarball_url, integrity_hash, tarball_dir; ignore_existence=true, verbose=true)

    lib_dir = joinpath(tarball_dir, "clFFT-$(version)-Linux-x64/lib64")
    libname = "libclFFT.so"
end

if Sys.iswindows()
    tarball_url = "$base_url-Windows-x64.zip"
    integrity_hash = "737ba79dba57e025e72586e06cb1b7906f40e6a4a1b0e390516aabc53546eb9c"
    tarball_path = joinpath(tarball_dir, "clFFT-$(version)-Windows-x64.zip")

    download_verify(tarball_url, integrity_hash, tarball_path; force=true, verbose=true)
    InfoZIP.unzip(tarball_path, joinpath(tarball_dir))    

    lib_dir = joinpath(tarball_dir, "clFFT-$(version)-Windows-x64/lib64/import/")
    libname = "clFFT.lib"
end

products = Product[
    LibraryProduct(lib_dir, libname, :libclfft)
]

# Write out a deps.jl file
write_deps_file(joinpath(@__DIR__, "deps.jl"), products, verbose=true)
