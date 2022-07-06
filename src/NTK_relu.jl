using DrWatson
@quickactivate "decoding_cneuro"
using KernelFunctions
using KernelFunctions:validate_inputs
import KernelFunctions: kernelmatrix

struct NTKRelu <: Kernel end

function (κ::NTKRelu)(x, y)
    validate_inputs(x,y)
    xn,yn = norm(x),norm(y)
    u = clamp(dot(x,y)/(xn*yn),-1,1)
    return xn*yn*(u(1-acos(u)/π) + (√(1-u^2))/π)
end

function kernelmatrix(::NTKRelu, x::ColVecs, y::ColVecs)
    validate_inputs(x, y)
    XN,YN = norm.(x),norm.(y)
    u = clamp.((x.X'*y.X)./(XN*YN'),-1,1)
    return (XN*YN').*(2*u.*(1 .-acos.(u)/π) .+ sqrt.(1 .-u.^2)/π )
end

function kernelmatrix(::NTKRelu, x::ColVecs)
    XN = norm.(x)
    u = clamp.((x.X'*x.X)./(XN*XN'),-1,1)
    return (XN*XN').*(2*u.*(1 .-acos.(u)/π) .+ sqrt.(1 .-u.^2)/π )
end

Base.show(io::IO, ::NTKRelu) = print(io, "Neural Network Kernel Relu")