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


scatter(σVec,mean(ε_last),yaxis=:log10,markersize=6,
     c=c1',legend=:none)
plot!(σVec,2*mean(ε_id),yaxis=:log10,legend=:none,c=c1')
##
#optimal quantities
εo = [findmin.(ε_last,dims=1)[n][1] for n=1:8]
εo_id = [findmin.(ε_id,dims=1)[n][1] for n=1:8]
plot(NVec,mean(εo)',yaxis=:log10)
plot!(NVec,mean(εo_id)',yaxis=:log10)

##
#Covariance matrices
λ_dec = [[pDec[i,j][n][:prob_dec].layers[1].W  
    for i = 1:length(σVec),j=1:length(NVec)] for n=1:8]
Vs = [[pDec[i,j][n][:tc]
    for i = 1:length(σVec),j=1:length(NVec)] for n=1:8]

Dcov,ODcov = [zeros(length(σVec),length(NVec)) for n=1:2]
for n=1:8
    for i=1:length(σVec), j=1:length(NVec)
        M = cov(λ_dec[n][i,j],Vs[n][i,j]')
        Dcov[i,j] += mean(diag(M))
        ODcov[i,j] += mean(triu(M,1)[triu(M,1) .!=0])
    end
end
Dcov ./=8
ODcov ./=8