using DrWatson
@quickactivate "decoding_cneuro"
#Strutcture and function for 1D model and ML-MSE inference
#using TensorBoardLogger,Logging
#using Plots
using JLD2
include(srcdir("kernel_utilities.jl"))
include(srcdir("utils.jl"))
include(srcdir("probabilistic_decoder.jl"))
#include(srcdir("plot_utils.jl"))
#plotlyjs(size=(400,300))
##

##
function prob_decoder(N::Int,σi; MaxEpochs=2000,nets=8,bs=50,opt=ADAM)
    pdecDicts = Dict()
    t = ScaleTransform(1/(sqrt(2)*σi))
    #Kernel matrices
    K = kernelmatrix(k,t(vcat(x_samples,x_m)))
    #Parameters to datapoint ratio
    P = M*N*γ
    @info N,P,σi
    Threads.@threads for n = 1:nets
        pdecD = Dict()
        #Generate data
        V = mvn_sample(K,N);
        V_m = V[:,end-M+1:end] 
        idx_trn = rand(1:length(x_samples),P)
        x_trn = x_samples[idx_trn]
        R_trn = V[:,idx_trn] + sqrt(η)*randn(N,P)
        idx_tst = rand(1:length(x_samples),n_tst)
        x_tst = x_samples[idx_tst]
        R_tst = V[:,idx_tst] + sqrt(η)*randn(N,n_tst)
        #Format data for probabilistic_decoder
        data,x_m = one_hotdataset(x_trn,x_tst,R_trn,R_tst,M,bs=bs)
        #Train decoder and computes ideal error
        p_dec, history = train_prob_decoder(data,x_m,x_tst,M=M,
            epochs=MaxEpochs,opt= opt)
        ε_id = mse_ideal(V_m,η,x_m,R_tst,x_tst')
        #Store: pecoder, ideal error, history of training and tuning curves
        pdecD[:ε_id] =ε_id
        pdecD[:prob_dec] = p_dec
        pdecD[:history] = history
        pdecD[:tc] = V_m
        pdecDicts[:($n)] = pdecD
        @info "Finished" n "on thread" Threads.threadid()
    end
    return pdecDicts
end
##
nets =8   #Numbers of test networks
#Dataset parameters
P_i= 2000  #How many I can sample from (not so important)
n_tst = Int.(1e5)
γ = 3 #Parameter/datapoint ratio
x_samples = sort((rand(Float32,P_i).-0.5f0))
#Ideal decoder parameters
M= 500
bin = range(-0.5,0.5,length=M+1)
x_m = bin[1:end-1] .+ diff(bin)/2
# Network parameters
σVec = (5:7:54)/500
NVec = 20:20:100
k =SqExponentialKernel()
η = 0.1
## Run simulations
pdec  = [prob_decoder(N,σi) for σi=σVec,N=NVec]
#Save results
##
Nmin,Nmax = first(NVec),last(NVec)
name = savename("prob_dec" , (@dict Nmin Nmax η γ),"jld2")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"prob_decoder" => pdec)
safesave(datadir("sims/probabilistic_decoder",name) ,data)




