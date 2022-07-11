using DrWatson
@quickactivate "decoding_cneuro"
#using TensorBoardLogger,Logging
using Plots
using StatsBase,Distributions,LinearAlgebra,Flux
include(srcdir("plot_utils.jl"))
##
mydir= datadir("sims/probabilistic_decoder")
flnames = filter(x ->occursin(r"prob_dec",x),
    readdir(mydir))
myf = flnames[3]
data = load(datadir(mydir,myf))
γ_e = split(myf,('_'))[findfirst(occursin.("γ",split(myf,'_')))] 
γ = eval.(Meta.parse.(γ_e))
η = 0.3
σVec = data["σVec"]
NVec = data["NVec"]
pDec = data["prob_decoder"]
##
ε_last = [[last(pDec[(σi,N)]["$n"][:history][:mse_tst]) 
    for σi = σVec,N= NVec] for n=1:8]
ε_min = [[minimum(pDec[(σi,N)]["$n"][:history][:mse_tst]) 
    for σi = σVec,N= NVec] for n=1:8]
ε_id = [[last(pDec[(σi,N)]["$n"][:ε_id]) 
    for σi = σVec,N= NVec] for n=1:8]
##
c1 = C(cgrad(:viridis),length(NVec))


p1 = plot(σVec,mean(ε_last),yaxis=:log10,markersize=6,
     c=c1',legend=:none,m=:o,xlabel=L"$\sigma$",ylabel = L"$\varepsilon^2$")
name = savename("evssigma" , (@dict  η γ),"svg")
#safesave(plotsdir("prob_dec",name) ,p1)
#plot!(σVec,mean(ε_id),yaxis=:log10,legend=:none,c=c1')

##
yt = ([10^(0) ,2, 10^(1),50 ],["1","2","10","50"])
p2=plot(σVec,mean(ε_min)./mean(ε_id),yaxis=:log10,markersize=6,
     c=c1',legend=:none,m=:o,yticks=yt,xlabel=L"$\sigma$",
     ylabel = L"$\varepsilon^2/\varepsilon_{id}^2$")
name = savename("evseid" , (@dict  η γ),"svg")
#safesave(plotsdir("prob_dec",name) ,p2)
##
σo = [[σVec[I[1]] for I  = findmin.(ε_last,dims=1)[n][2]] for n=1:8]
σo_id = [[σVec[I[1]] for I  = findmin.(ε_id,dims=1)[n][2]] for n=1:8]

p3 = plot(NVec,[mean(σo)',mean(σo_id)'],m=:o,label=["B-Dec" "Id" ],
    markersize=6,linewidth=2,ylabel=L"$\sigma^*$",xlabel=L"$N$")
name = savename("sigmao" , (@dict  η γ),"svg")
#safesave(plotsdir("prob_dec",name) ,p3)
#optimal quantities
p4 = εo = [findmin.(ε_min,dims=1)[n][1] for n=1:8]
εo_id = [findmin.(ε_id,dims=1)[n][1] for n=1:8]
p4=plot(NVec,[mean(εo)',mean(εo_id)'],yaxis=:log10,m=:o,label=["B-Dec" "Id" ],
    markersize=6,linewidth=2,ylabel=L"$\varepsilon^2(\sigma^*)$",xlabel=L"$N$")
name = savename("eo" , (@dict  η γ),"svg")
#safesave(plotsdir("prob_dec",name) ,p4)

##
#Covariance matrices
c2 = C(cgrad(:magma),length(NVec))
λ_dec = [[pDec[(σi,N)]["$n"][:prob_dec].layers[1].W  
    for σi = σVec,N=NVec] for n=1:8]
Vs = [[pDec[(σi,N)]["$n"][:tc]
    for σi = σVec,N=NVec] for n=1:8]

Dcov,ODcov,Dcv,ODv = [zeros(length(σVec),length(NVec)) for n=1:4]

for i=1:length(σVec), j=1:length(NVec)
    D,OD = [],[]
    for n=1:8
        M = cor(λ_dec[n][i,j]',Vs[n][i,j]/η)
        push!(D,vec(diag(M)))
        push!(OD,vec(triu(M,1)[triu(M,1) .!=0]))
    end
    D = vcat(D...)
    OD = vcat(OD...)
    Dcov[i,j] = mean(D)
    Dcv[i,j]  = std(D)
    ODcov[i,j] = mean(OD)
    ODv[i,j] = std(OD)
end
transl = hcat([i*ones(length(σVec)) for i=-0.003:0.0015:0.003]...)
p5 = scatter(σVec.+transl,Dcov,c=c1',yerr = Dcv',
    xlabel = L"$\sigma$",ylabel = L"$\rho$",markersize=6,legend=:none)
scatter!(p5,σVec.+transl,ODcov,c=c1',yerr = ODv',
    xlabel = L"$\sigma$",ylabel = L"$\rho$",markersize=6,legend=:none)
name = savename("Vlambdacov" , (@dict  η γ),"svg")
safesave(plotsdir("prob_dec",name) ,p5)