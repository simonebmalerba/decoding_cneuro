using DrWatson
@quickactivate "decoding_cneuro"
using KernelFunctions
using KernelFunctions:validate_inputs
import KernelFunctions: kernelmatrix
##Overwrite methods for NeuralTangentKernel by clipping the argument of 
#asin()
function (κ::NeuralNetworkKernel)(x, y)
    return asin(clamp(dot(x, y) / sqrt((1 + sum(abs2, x)) * (1 + sum(abs2, y))),-1,1))
end

function kernelmatrix(::NeuralNetworkKernel, x::ColVecs, y::ColVecs)
    validate_inputs(x, y)
    X_2 = sum(x.X .* x.X; dims=1)
    Y_2 = sum(y.X .* y.X; dims=1)
    XY = x.X' * y.X
    return asin.(clamp.(XY ./ sqrt.((X_2 .+ 1)' * (Y_2 .+ 1)),-1,1))
end

function kernelmatrix(::NeuralNetworkKernel, x::ColVecs)
    X_2_1 = sum(x.X .* x.X; dims=1) .+ 1
    XX = x.X' * x.X
    return asin.(clamp.(XX ./ sqrt.(X_2_1' * X_2_1),-1,1))
end