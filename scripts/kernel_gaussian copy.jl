##
using DrWatson
@quickactivate "decoding_cneuro"
#Strutcture and function for 1D model and ML-MSE inference
#using TensorBoardLogger,Logging
include(srcdir("kernel_utilities.jl"))
include(srcdir("plot_utils.jl"))
include(srcdir("utils.jl"))
plotlyjs(size=(400,300))
##
#c = C(cgrad(:viridis),N)
function lower_bound(N,σi,P)
    #Compute lower bound through anayltical expression
    t = ScaleTransform(1/(sqrt(2)*σi))
    #Ideal tiling of size P
    x_m_ext = range(-0.5,0.5,length=P)
    K_id = kernelmatrix(k,t(x_m_ext))
    #Compute ideal integral problem: deterministic lowr bound
    F = eigen(K_id) 
    d̄ = reverse(abs.((F.values/P)))
    φ = F.vectors'
    w̄ = reverse(φ*x_m_ext)
    d_approx = P*d̄.+η
    λ_e, dλ_e = effective_ridge0(d_approx,N)
    N_e = N-sum((d_approx./(d_approx .+ λ_e)).^2)
    η_e = η*sum((d_approx.*w̄.^2)./(d_approx .+ λ_e).^2)
    return η_e,N_e
end

function linear_decoder(N::Int,σi; nets = 8)
    lDicts = Dict()
    t = ScaleTransform(1/(sqrt(2)*σi))
    P = γ*N
    @info N,P,σi
    x_trn = sort((rand(Float32,P).-0.5f0))
    K_id = kernelmatrix(k,t(vcat(x_trn,x_m)))
    n_tste = Int(round(n_tst/P))
    idx_tst = repeat(1:P,n_tste)
    x_tst = repeat(x_trn,n_tste)
    for n = 1:nets
        lD = Dict()
        V = mvn_sample(K_id,N);
        V_m = V[:,end-M+1:end] 
        R_trn = V[:,1:P] + sqrt(η)*randn(N,P)
        #Sample more data points for testing
        R_tst = V[:,idx_tst] + sqrt(η)*randn(N,length(idx_tst))
        #Decoding weights
        w = (R_trn*R_trn')\(R_trn*x_trn)
        b = w'*mean(R_trn,dims=2) .- mean(x_trn) 
        #Decoder output
        x_ext = R_tst'w  .+ b
        x_Matrix = reshape(x_ext,P,:)
        ε =  mean((x_Matrix .- x_trn).^2)
        x_ext_av = mean(x_Matrix,dims=2)
        B = mean((x_ext_av - x_trn).^2)
        V2 = mean(var(x_Matrix,dims=2,corrected=false))
        ε_id = mse_ideal(V_m,η,x_m,R_tst,x_tst')
        lD[:ε_id] = ε_id
        lD[:mse] = ε
        lD[:w] = w
        lD[:tc] = V_m
        lD[:bias] = B
        lD[:V2] = V2
        lD[:f_av] = x_ext_av
        lDicts[:($n)] = lD
    end

    lDicts[:lb] = lower_bound(N,σi,P)
    return lDicts
end
##
nets=8
#Dataset parameters
n_tst = Int.(1e5)
γ = 10
#Network parameters
#N = 100
#σi = 30/500
k = SqExponentialKernel()
η = 0.1
M=500
##
bin = range(-0.5,0.5,length=M+1)
x_m = bin[1:end-1] .+ diff(bin)/2
##
σVec = (5:10:55)/500
NVec = 60:20:200

linDec = [linear_decoder(N,σi,nets=1) for σi = σVec, N=NVec]


##
Nmin,Nmax = first(NVec),last(NVec)
name = savename("lin_dec" , (@dict Nmin Nmax η γ),"jld2")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"linDec" => linDec)
safesave(datadir("sims/linear_decoder",name) ,data)

