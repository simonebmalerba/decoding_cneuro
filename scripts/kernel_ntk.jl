##
using DrWatson
@quickactivate "decoding_cneuro"
#Strutcture and function for 1D model and ML-MSE inference
#using TensorBoardLogger,Logging
include(srcdir("kernel_utilities.jl"))
#include(srcdir("plot_utils.jl"))
include(srcdir("utils.jl"))
include(srcdir("NTK_relu.jl"))
include(srcdir("NeuralNetworkKernel_clipped.jl"))

#plotlyjs(size=(400,300))
##
#c = C(cgrad(:viridis),N)

function ntk_decoder(N::Int,σi; nets = 8,rich=0)
    ntkDicts = Dict()
    t = ScaleTransform(1/(sqrt(2)*σi))
    P= γN*N
    @info N,σi,P
    x_trn = sort((rand(Float32,P).-0.5f0))
    K_id = kernelmatrix(k,t(vcat(x_trn,x_m)))
    n_tste = Int(round(n_tst/P))
    idx_tst = repeat(1:P,n_tste)
    x_tst = repeat(x_trn,n_tste)
    ntkDicts = Dict("$n"=> Dict() for n=1:nets)
    for n = 1:nets
        ntkD = ntkDicts["$n"]
        V = mvn_sample(K_id,N);
        V_m = V[:,end-M+1:end] 
        R_trn = V[:,1:P] + sqrt(η)*randn(N,P)
        #Sample more data points for testing
        R_tst = V[:,idx_tst] + sqrt(η)*randn(N,length(idx_tst))
        #Kernel regression solution
        #k_ntk = NTKRelu()
        k_ntk = NeuralNetworkKernel()
        K_NTK = kernelmatrix(k_ntk,R_trn)
        k_r = kernelmatrix(k_ntk,R_trn,R_tst)
        α = K_NTK\x_trn
        x_ext = (α'*k_r)'
        ε = mean((x_ext-x_tst).^2)
        @info ε
        #Decoder output
        ε_id = mse_ideal(V_m,η,x_m,R_tst,x_tst')
        ntkD[:ε_id] = ε_id
        ntkD[:mse] = ε
        ntkD[:tc] = V_m
    end
    return ntkDicts
end
##
nets=2
#Dataset parameters
n_tst = Int.(1e5)
γN = 300
#Network parameters
#N = 100
#σi = 30/500
k = SqExponentialKernel()
η = 0.1
M=500
##
bin = range(-0.5,0.5,length=M+1)
x_m = bin[1:end-1] .+ diff(bin)/2
##

σVec = ((5:8:55)/500)[2:4]
#NVec = 60:20:200
#σi = 20/500
NVec = 50:10:50
ntkDec = Dict((σi,N) => ntk_decoder(N,σi,nets=nets) for σi = σVec,N=NVec)
##
Nmin,Nmax = first(NVec),last(NVec)
name = savename("ntkErf_dec" , (@dict Nmin Nmax η γN),"jld2")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"ntkDec" => ntkDec)
safesave(datadir("sims/ntk_decoder",name) ,data)

