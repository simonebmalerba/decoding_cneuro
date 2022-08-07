using DrWatson
@quickactivate "decoding_cneuro"
#using TensorBoardLogger,Logging
using Plots
using StatsBase,Distributions
include(srcdir("plot_utils.jl"))
##
mydir= datadir("sims/dnn_decoder")
flnames = filter(x ->occursin(r"Mcomp",x),
    readdir(mydir))
myf = flnames[2]
data = load(datadir(mydir,myf))
σVec = data["σVec"]
MVec = data["MVec"]
#linDec = data["linDec"]
dnnDec = data["dnn_decoder"]
η = 0.3
γ = 3
##
ε = [[last(dnnDec[(σi,n)][:dnn][M][:history][:mse_tst]) 
    for σi = σVec,M=MVec] for n=1:4]
#lb = mean([[/(linDec[(σi,N)][n][:lb]...) for σi = σVec,N=NVec] for n=1:8])
ε_id = [[dnnDec[(σi,n)][:ε_id] for σi = σVec] for n=1:4]
c1 = C(cgrad(:viridis),length(MVec))
#p1 = plot(size=(400,300))
#yt = ([10^(-3.5) , 10^(-3), 10^(-2.5), 10^(-2) ],["0.0003","0.001","0.003","0.01"])
p1 = plot(σVec,mean(ε),ribbon=std(ε), xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    m=:o,linewidth=1,linestyle=:dash,markersize=6,yaxis=:log10,c=c1[4:end]',legend=:none)#)
plot!(p1,σVec,mean(ε_id), xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    m=:o,linewidth=1,linestyle=:dash,markersize=6,yaxis=:log10,c=c1[1:3]',legend=:none)#)
##Error curves asa function of σ
#plot!(p1,σVec,lb,yaxis=:log10,c=c1',linewidth=1,linealpha=0.7)

plot!(p1,σVec,mean(ε_id), xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    markersize=6,linewidth=1,m=:diamond,yaxis=:log10,c=c1[1:3]')

name = savename("evssigma_dnn1000" , (@dict  η γ),"svg")
safesave(plotsdir("dnn_dec",name) ,p1)
##
Δε = maximum(ε,dims=2) - minimum(ε_MLP,dims=2);
minε = minimum(ε_MLP,dims=2);
perc_diff = (ε_MLP .- minε) .< Δε*2/100
Msat = MVec[findfirst.(x -> x,eachrow(perc_diff))];