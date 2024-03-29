using DrWatson
@quickactivate "decoding_cneuro"
#using TensorBoardLogger,Logging
using Plots
using StatsBase,Distributions
include(srcdir("plot_utils.jl"))
##
#mydir= datadir("sims/linear_decoder")
#mydir= datadir("sims/ntk_decoder")
mydir = datadir("sims/ntk_decodervsrich")
flnames = filter(x ->occursin(r"ntkErf",x),
    readdir(mydir))
myf = flnames[10]
data = load(datadir(mydir,myf))
γ_e = split(myf,('_'))[findfirst(occursin.("γ",split(myf,'_')))] 
γ = eval.(Meta.parse.(γ_e))
σVec = data["σVec"]
NVec = data["NVec"]
#linDec = data["linDec"]
ntkDec = data["ntkDec"]
η =0.1
#γ = 20
##
ε = [[ntkDec[(σi,N)]["$n"][:mse] for σi = σVec,N=NVec] for n=1:2]
#lb = mean([[/(linDec[(σi,N)][n][:lb]...) for σi = σVec,N=NVec] for n=1:8])
ε_id = [[ntkDec[(σi,N)]["$n"][:ε_id] for σi = σVec,N=NVec] for n=1:2]
c1 = C(cgrad(:viridis),length(NVec))
#p1 = plot(size=(400,300))
#yt = ([10^(-3.5) , 10^(-3), 10^(-2.5), 10^(-2) ],["0.0003","0.001","0.003","0.01"])
p1 = plot(σVec,mean(ε),ribbon=std(ε), xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    m=:o,linewidth=1,linestyle=:dash,markersize=6,yaxis=:log10,c=c1[3:end]',legend=:none)#)
#plot!(p1,σVec,mean(ε), xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    m=:o,linewidth=1,linestyle=:dash,markersize=6,yaxis=:log10,c=c1',legend=:none)#)
##Error curves asa function of σ
#plot!(p1,σVec,lb,yaxis=:log10,c=c1',linewidth=1,linealpha=0.7)
plot!(p1,σVec,mean(ε_id), xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    markersize=6,linewidth=1,m=:diamond,yaxis=:log10,c=c1[3:end]')
name = savename("evssigma_Erf" , (@dict  η γ),"svg")
safesave(plotsdir("ntk_dec",name) ,p1)
##
p2 = plot(σVec,mean(ε)./mean(ε_id), xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
    m=:o,markersize=6,yaxis=:log10,legend=:none,c=c1')
##
B = mean([[linDec[(σi,N)][n][:bias] for σi = σVec,N=NVec] for n=1:8])
V2 = mean([[linDec[(σi,N)][n][:V2] for σi = σVec,N=NVec] for n=1:8])
V1 = [mean(var([linDec[(σi,N)][n][:f_av]  for n=1:8],corrected=false)) for σi = σVec,N=NVec]
p2 = plot(layout=(2,1))
plot!(p2[1],σVec, B./V2,c=c1',legend=:none,xaxis=:nothing,ylabel = L"$B/V_2$")
plot!(p2[2],σVec, V1./V2,c=c1',legend=:none,ylabel=L"$V_1/V_2$",xlabel=L"$\sigma$")
name = savename("BvsV1vsV2" , (@dict  η γ),"svg")
safesave(plotsdir("lin_dec",name) ,p2)
##
#Optimal error curves
#plot(NVec,mean(ε_id)', xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
#    markersize=6,yaxis=:log10,legend=:none,c=c1')
p3 = plot(xlabel=L"$N$",ylabel = L"$\varepsilon(\sigma^*)$")
#ε = [[linDec[(σi,N)][String("$n")][:mse] for σi = σVec,N=NVec] for n=1:8]
#lb = mean([[/(linDec[(σi,N)][String("$n")][:lb]...) for σi = σVec,N=NVec]
#     for n=1:8])
##
c2 = C(cgrad(:speed),5)
εo = [findmin.(ε,dims=1)[n][1] for n=1:8]
lbo = findmin(lb,dims=1)[1]
εo_id = [findmin.(ε_id,dims=1)[n][1] for n=1:8]
#σo = [[σVec[I[1]] for I  = findmin.(ε,dims=1)[n][2]] for n=1:8]
scatter!(p3,NVec,mean(εo)',m=:o,markersize=6,c=c2[3])
plot!(p3,NVec,lbo',c=c2[5],label=L"$\gamma=20$")
plot!(p3,NVec,mean(ε_id)[end,:])
name = savename("e_optvsN_gamma2_20" , (@dict  η ),"svg")
safesave(plotsdir("lin_dec",name) ,p3)
##
