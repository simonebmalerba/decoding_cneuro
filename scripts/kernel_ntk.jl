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

#function ntk_decoder(N::Int,σi; nets = 8)
    ntkDicts = Dict()
    t = ScaleTransform(1/(sqrt(2)*σi))
    @info N,σi
    x_trn = sort((rand(Float32,P).-0.5f0))
    K_id = kernelmatrix(k,t(vcat(x_trn,x_m)))
    n_tste = Int(round(n_tst/P))
    idx_tst = repeat(1:P,n_tste)
    x_tst = repeat(x_trn,n_tste)
    #for n = 1:nets
        ntkD = Dict()
        V = mvn_sample(K_id,N);
        V_m = V[:,end-M+1:end] 
        R_trn = V[:,1:P] + sqrt(η)*randn(N,P)
        #Sample more data points for testing
        R_tst = V[:,idx_tst] + sqrt(η)*randn(N,length(idx_tst))
        #Kernel regression solution
        k_ntk = NeuralNetworkKernel()
        K_NTK = kernelmatrix(k_ntk,R_trn)
        k_r = kernelmatrix(k_ntk,R_trn,R_tst)
        α = K_NTK\x_trn
        x_ext = (α'*k_r)'
        ε = mean((x_ext-x_tst).^2)
        @info ε
        #Decoder output
        ε_id = mse_ideal(V_m,η,x_m,R_tst,x_tst')
        ntkD[:ε_id] = ε_id
        ntkD[:mse] = ε
        ntkD[:tc] = V_m
        ntkDicts[:($n)] = ntkD
    end
    return ntkDicts
end
##
nets=8
#Dataset parameters
n_tst = Int.(1e5)
P = 5000
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
#NVec = 60:20:200
σi = 20/500
N = 60
ntkDec = [ntk_decoder(50,σi,nets=4) for σi = σVec]

##
Nmin,Nmax = first(NVec),last(NVec)
name = savename("lin_dec" , (@dict Nmin Nmax η γ),"jld2")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"linDec" => linDec)
safesave(datadir("sims/linear_decoder",name) ,data)

##
idx_trn = rand(1:length(x_trn),1500*N)
x_trn2 = x_trn[idx_trn]
R_trn = V[:,idx_trn] + sqrt(η)*randn(N,1500*N)
data_trn = Flux.Data.DataLoader((Float32.(R_trn),x_trn'),
            batchsize = bs,shuffle = true);
data_tst = Flux.Data.DataLoader((Float32.(R_tst),x_tst'),
    batchsize = 100,shuffle = false);
data = [data_trn,data_tst]
mydec = Chain(Dense(Float32.(sqrt(0.001/Md)*randn(Md,N)),zeros(Float32,Md),relu),
    Dense(Md,1,identity))
#Train decoder and computes ideal error
dec, history = train_dnn_dec(data,dec=mydec,
    epochs=2000,min_diff=1e-8)
##