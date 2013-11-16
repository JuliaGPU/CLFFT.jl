import CLFFT
using FactCheck 
using Base.Test

facts("Version") do 
    @fact isa(CLFFT.version(), NTuple{3,Int}) => true
    @fact CLFFT.version()[1] >= 2 => true
    @fact CLFFT.version()[2] >= 1 => true
    @fact CLFFT.version()[3] >= 0 => true
end


