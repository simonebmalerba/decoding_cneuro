using DrWatson
@quickactivate "decoding_cneuro"
#using TensorBoardLogger,Logging
using Plots
using StatsBase,Distributions
include(srcdir("plot_utils.jl"))
##
mydir= datadir("sims/probabilistic_decoder")
flnames = filter(x ->occursin(r"prob_dec",x),
    readdir(mydir))
myf = flnames[1]
data = load(datadir(mydir,myf))
γ_e = split(myf,('_'))[findfirst(occursin.("γ",split(myf,'_')))] 
γ = eval.(Meta.parse.(γ_e))
σVec = data["σVec"]
NVec = data["NVec"]
pDec = data["prob_decoder"]
##
ε_last = [[last(pDec[i,j][n][:history][:mse_tst]) 
    for i = 1:length(σVec),j=1:length(NVec)] for n=1:8]
ε_min = [[minimum(pDec[i,j][n][:history][:mse_tst]) 
    for i = 1:length(σVec),j=1:length(NVec)] for n=1:8]
ε_id = [[last(pDec[i,j][n][:ε_id]) 
    for i = 1:length(σVec),j=1:length(NVec)] for n=1:8]
##
c1 = C(cgrad(:viridis),length(NVec))

plot(σVec,mean(ε_min)./mean(ε_id),ribbon= std(ε_min),  c=c1',legend=:none,ylim=(0,10))
plot!(σVec,mean(ε_id),yaxis=:log10,legend=:none,c=c1')
##
εo = [findmin.(ε_last,dims=1)[n][1] for n=1:8]
εo_id = [findmin.(ε_id,dims=1)[n][1] for n=1:8]
plot(NVec,mean(εo)',yaxis=:log10)
plot!(NVec,mean(εo_id)',yaxis=:log10)