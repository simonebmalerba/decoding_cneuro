using DrWatson
@quickactivate "decoding_cneuro"
#Strutcture and function for 1D model and ML-MSE inference
#using TensorBoardLogger,Logging
using Plots,LaTeXStrings,KernelFunctions, Roots
include(srcdir("kernel_utilities.jl"))
plotlyjs(size=(400,300))
f1 = Plots.font("sans-serif",12)
f2 = Plots.font("sans-serif",16)
fs = Dict(:guidefont => f2, :legendfont => f1, :tickfont => f1)
default(; fs...)
##
σVec = Float32.((5:10:55)/500)
##
N=50
η = 0.1
P = 2000
ntst = 3000
x_trn = sort((rand(Float32,P).-0.5f0))
x_tst = sort((rand(Float32,ntst).-0.5f0))
σi  = 20/500
k =SqExponentialKernel()#
k̃ = k + η*WhiteKernel()
t = ScaleTransform(1/(sqrt(2)*σi))
#Kernel matrices
K̃ = kernelmatrix(k̃,t(vcat(x_trn,x_tst)))
K̃_trn = K̃[1:P,1:P]
R = mvn_sample(K̃,N)
R_trn = R[:,1:P];
R_tst = R[:,P+1:end]
k_n = NeuralNetworkKernel()
K_ntk = kernelmatrix(k_n,R_trn)
k_nx = kernelmatrix(k_n,R_trn,R_tst)
x_n = x_trn'inv(K_ntk)*k_nx
e = mean((x_n' - x_tst).^2)
M=500
bin = range(-0.5,0.5,length=M+1)
bidx = searchsortedfirst.(Ref(bin),vcat(x_trn,x_tst)).-1
V = hcat([mean(R[:,bidx.==n],dims=2) for n=1:maximum(bidx)]...)
K_0 = kernelmatrix(k_n,V)