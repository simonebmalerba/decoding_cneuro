using DrWatson
@quickactivate "decoding_cneuro"
#using TensorBoardLogger,Logging
using Plots
using StatsBase,Distributions
include(srcdir("plot_utils.jl"))
##
#mydir= datadir("sims/linear_decoder")
mydir= datadir("sims/ntk_decoder")
flnames = filter(x ->occursin(r"ntkErf",x),
    readdir(mydir))
myf = flnames[9]
η =0.1
p4 = plot(xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",yaxis=:log10)
c2 = C(cgrad(:speed),5)
i=1
for f in vcat(flnames[3],flnames[6:end])
    data = load(datadir(mydir,f))
    γ_e = split(f,('_'))[findfirst(occursin.("γ",split(f,'_')))] 
    γ = eval.(Meta.parse.(γ_e))
    σVec = data["σVec"]
    NVec = data["NVec"]
    #linDec = data["linDec"]
    ntkDec = data["ntkDec"]
    #γ = 20
    ##
    nets = length(ntkDec[(σVec[1],NVec[1])])
    ε = [[ntkDec[(σi,N)]["$n"][:mse] for σi = σVec,N=NVec] for n=1:nets]
    #lb = mean([[/(linDec[(σi,N)][n][:lb]...) for σi = σVec,N=NVec] for n=1:8])
    ε_id = [[ntkDec[(σi,N)]["$n"][:ε_id] for σi = σVec,N=NVec] for n=1:nets]
    #Different P for N=50, NTK decoder
    P= γ*50
    if i==1
        plot!(p4,σVec,mean(ε_id)[:,end],xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$",
        m=:diamond,linewidth=2,markersize=6,yaxis=:log10,
            c=1,
            label="Id")
    end
    plot!(p4,σVec,mean(ε)[:,end],
        m=:o,linewidth=2,markersize=6,c=c2[i]',label=L"$P=%$P$")
    i +=1
end
p4
name = savename("evssigma_highP" , (@dict  η),"svg")
safesave(plotsdir("ntk_dec",name) ,p4)
##
for σi =σVec
    ntkDec[(σi,50)]["5"] = ntkDec3[(σi,50)]["1"]
    ntkDec[(σi,50)]["6"] = ntkDec3[(σi,50)]["2"]
end