using DrWatson
@quickactivate "decoding_cneuro"
#using TensorBoardLogger,Logging
using Plots
using StatsBase,Distributions
include(srcdir("plot_utils.jl"))
##
mydir = datadir("sims/ntk_decodervsrich")
flnames = filter(x ->occursin(r"ntkwrich",x),
    readdir(mydir))
myf = flnames[1]
data = load(datadir(mydir,myf))
γ_e = split(myf,('_'))[findfirst(occursin.("γ",split(myf,'_')))] 
γ = eval.(Meta.parse.(γ_e))
σVec = data["σVec"]
#linDec = data["linDec"]
ntkDec = data["ntkDec"]
η =0.3
#γ = 20
##
ε_ntk = [[ntkDec[(σi)]["$n"][:mse] for σi = σVec] for n=1:1]
ε_dnn = [[last(ntkDec[(σi)]["$n"][:history_rich][:mse_tst]) 
    for σi = σVec] for n=1:1]
ε_id = [[ntkDec[(σi)]["$n"][:ε_id] for σi = σVec] for n=1:4]
#p1 = plot(size=(400,300))
#yt = ([10^(-3.5) , 10^(-3), 10^(-2.5), 10^(-2) ],["0.0003","0.001","0.003","0.01"])
p1 = plot(σVec,mean(ε_ntk),ribbon=std(ε), xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    m=:o,linewidth=1,linestyle=:dash,markersize=6,yaxis=:log10,label="NTK-Dec")#)
plot!(p1,σVec,mean(ε_dnn), ribbon=std(ε),xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    m=:o,linewidth=1,linestyle=:dash,markersize=6,yaxis=:log10,label="dnn (rich)-Dec")#)
#plot!(p1,σVec,lb,yaxis=:log10,c=c1',linewidth=1,linealpha=0.7)
plot!(p1,σVec,mean(ε_id), xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    markersize=6,linewidth=1,m=:diamond,yaxis=:log10,label="Id")
name = savename("evssigma_ntkvsrich" , (@dict  η γ),"svg")
#safesave(plotsdir("ntk_dec",name) ,p1)
##
ntrn = length(ntkDec[(0.058)]["1"][:history_rich][:ΔP])
ΔP = [ntkDec[(0.058)]["1"][:history_rich][:ΔP][i][1] for i =1:ntrn]
log.(norm.(ΔP))
p2 = 
