using DrWatson
@quickactivate "decoding_cneuro"
#Strutcture and function for 1D model and ML-MSE inference
#using TensorBoardLogger,Logging
#using Plots
using JLD2
include(srcdir("kernel_utilities.jl"))
include(srcdir("utils.jl"))
include(srcdir("dnn_decoder.jl"))
#include(srcdir("plot_utils.jl"))
#plotlyjs(size=(400,300))
##

##
function dnn_dec(σi; MaxEpochs=2000,bs=128,opt=ADAM)
    t = ScaleTransform(1/(sqrt(2)*σi))
    #Kernel matrices
    K = kernelmatrix(k,t(vcat(x_samples,x_m)))
    #Parameters to datapoint ratio
    ddecDicts = Dict(Md=> Dict() for M=1:MVec)
    idx_tst = rand(1:length(x_samples),n_tst)
    x_tst = x_samples[idx_tst]
    R_tst = V[:,idx_tst] + sqrt(η)*randn(N,n_tst)
    data_tst = Flux.Data.DataLoader((Float32.(R_tst),x_tst'),
            batchsize = 512,shuffle = false);
    #Generate data
    V = mvn_sample(K,N);
    V_m = V[:,end-M+1:end] 
    ddecDicts[:tc] = V_m
    #Ideal error
    ε_id = mse_ideal(V_m,η,x_m,R_tst,x_tst')
    #Store: pecoder, ideal error, history of training and tuning curves
    ddecDicts[:ε_id] =ε_id
    for Md=MVec
        ddecD = ddecDicts[Md]
        P = Md*N*γ
        @info Md,P,σi
        idx_trn = rand(1:length(x_samples),P)
        x_trn = x_samples[idx_trn]
        R_trn = V[:,idx_trn] + sqrt(η)*randn(N,P)
        #Format data for probabilistic_decoder
        data_trn = Flux.Data.DataLoader((Float32.(R_trn),x_trn'),
            batchsize = bs,shuffle = true);
        data = [data_trn,data_tst]
        #Initialize decoder architecture
        mydec = Chain(Dense(Float32.(sqrt(α/Md)*randn(Md,N)),zeros(Float32,Md),relu),
            Dense(Md,1,identity))
        #Train decoder and computes ideal error
        dec, history = train_dnn_dec(data,dec=mydec,
            epochs=MaxEpochs,min_diff=1f-7)
        ddecD[:dnn_dec] = dec
        ddecD[:history] = history
        @info "Finished" Md  
    end
    return ddecDicts
end
##
nets =8   #Numbers of test networks
#Dataset parameters
P_i= 2000  #How many I can sample from (not so important)
n_tst = Int.(1e5)
γ = 3 #datapoint/parameter ratio

x_samples = sort((rand(Float32,P_i).-0.5f0))
#Ideal decoder parameters
M= 500
bin = range(-0.5,0.5,length=M+1)
x_m = bin[1:end-1] .+ diff(bin)/2
##
#Decoding parameters
α = 0.001 #Variance of weights at initialization
# Network parameters
σVec = (5:8:54)/500
N =  50
MVec = Int.(round.(10 .^(range(1,3,length=7))))

k =SqExponentialKernel()
η = 0.3
## Run simulations
d_dec  = Dict((σi,n) => dnn_dec(σi) for σi = σVec, n=1:nets)
#Save results
##
Nmin,Nmax = first(NVec),last(NVec)
name = savename("dnn_dec" , (@dict Nmin Nmax γ η Md α),"jld2")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"dnn_decoder" => d_dec)
safesave(datadir("sims/dnn_decoder",name) ,data)