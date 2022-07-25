using DrWatson
@quickactivate "decoding_cneuro"
#using TensorBoardLogger,Logging
using Plots
using StatsBase,Distributions
include(srcdir("plot_utils.jl"))
##
mydir= datadir("sims/dnn_decoder")
flnames = filter(x ->occursin(r"dnn_dec",x),
    readdir(mydir))
myf = flnames[3]
data = load(datadir(mydir,myf))
σVec = data["σVec"]
NVec = data["NVec"]
#linDec = data["linDec"]
dnnDec = data["dnn_decoder"]
η = 0.3
γ = 3
##
ε = [[last(dnnDec[(σi,N)]["$n"][:history][:mse_tst]) 
    for σi = σVec,N=NVec] for n=1:4]
#lb = mean([[/(linDec[(σi,N)][n][:lb]...) for σi = σVec,N=NVec] for n=1:8])
ε_id = [[dnnDec[(σi,N)]["$n"][:ε_id] for σi = σVec,N=NVec] for n=1:4]
c1 = C(cgrad(:viridis),length(NVec)+3)
#p1 = plot(size=(400,300))
#yt = ([10^(-3.5) , 10^(-3), 10^(-2.5), 10^(-2) ],["0.0003","0.001","0.003","0.01"])
p1 = plot(σVec,mean(ε),ribbon=std(ε), xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    m=:o,linewidth=1,linestyle=:dash,markersize=6,yaxis=:log10,c=c1[4:end]',legend=:none)#)
##Error curves asa function of σ
#plot!(p1,σVec,lb,yaxis=:log10,c=c1',linewidth=1,linealpha=0.7)

plot!(p1,σVec,mean(ε_id), xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    markersize=6,linewidth=1,m=:diamond,yaxis=:log10,c=c1[4:end]')

name = savename("evssigma_dnn1000" , (@dict  η γ),"svg")
safesave(plotsdir("dnn_dec",name) ,p1)

yt = ([10^(0) ,2, 10^(1),50,100 ],["1","2","10","50","100"])
p2= plot(σVec,mean(ε)./mean(ε_id), xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    markersize=6,linewidth=2,yaxis=:log10,yticks=yt,c=c1[4:end]')
name = savename("eratio_dnn1000" , (@dict  η γ),"svg")
safesave(plotsdir("dnn_dec",name) ,p2)