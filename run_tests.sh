#! /bin/sh

if [[ :$LD_LIBRARY_PATH: == *:"/home/jake/opt/clFFT/build/library":* ]] ; then
	echo "CLFFT LIBRARY ON PATH"
else
    # oops, the directory is not on the path
    export LD_LIBRARY_PATH='/home/jake/opt/clFFT/build/library':${LD_LIBRARY_PATH}
fi

julia -L src/CLFFT.jl test/runtests.jl
