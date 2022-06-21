using DrWatson
@quickactivate "Random_coding"
using Flux
using Parameters: @with_kw
using ProgressMeter: Progress, next!
##
@with_kw mutable struct DataArgs
    ntrn = 50        #Number of noisy samples for the same stimulus x train
    ntst = 50         #Number of noisy samples for the same stimulus x test
    mb = 100           #Size of minibatches
    shuff = true      #Shuffle data
end

@with_kw mutable struct TrainArgsDnn
    lr = 1e-3              # learning rate
    epochs = 50             # number of epochs
    M = 500                 # latent dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    opt = ADAM             #Optimizer
    f = relu               ##Specify non linearity
end
##
function GaussianDataset(V::AbstractMatrix,η::AbstractFloat,
        x_test::AbstractArray; kws...)
    #Input: matrix of tuning curves in the shape N x L
    N,L = size(V)
    data_args = DataArgs(;kws...)
    R_trn = hcat([V .+ sqrt(η)*randn(Float32,N,L) for n=1:data_args.ntrn]...)
    x_trn = hcat([x_test' for n=1:data_args.ntrn]...)
    @info "Training points: $(data_args.ntrn) ,Test points: $(data_args.ntst)"
    R_tst = hcat([V .+ sqrt(η)*randn(Float32,N,L) for n=1:data_args.ntst]...)
    x_tst = hcat([x_test' for n=1:data_args.ntst]...)
    data_trn = Flux.Data.DataLoader((R_trn,x_trn),
        batchsize = data_args.mb,shuffle = data_args.shuff);
    data_tst = Flux.Data.DataLoader((R_tst,x_tst));
    data = [data_trn,data_tst]
    return data, data_args
end

## Mean squared error
mse_loss(dec,r,x) = Flux.mse(dec(r) , x)

function train_dnn_dec(data; dec = nothing, kws...)
    #Train DNN decoder on linear data
    data_trn,data_tst = data[1],data[2]
    N,mb = size(first(data_trn)[1])
    args = TrainArgsDnn(; kws...)
    if isnothing(dec)
        dec = Chain(Dense(N,args.M,args.f),
            Dense(args.M,1,identity))
    end
    #Optmizer for gradient descent
    opt = args.opt(args.lr)
    ps = Flux.params(dec)
    initp = deepcopy(Flux.params(dec))
    #Save training history
    history = Dict(:mse_trn => Float32[],:mse_tst => Float32[],:ΔP => [])
    trn_step =0
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(data_trn))
        l_av = 0
        count = 0
        for d in data_trn
            l, back = Flux.pullback(ps) do
                mse_loss(dec,d...)
            end
            #Running average
            l_av += l
            grad = back(1f0)
            count += 1
            Flux.Optimise.update!(opt, ps, grad)
            next!(progress; showvalues=[(:loss, l_av/count),(:epoch,epoch)])
            if trn_step % args.verbose_freq == 0
                push!(history[:mse_trn],l_av/count)
            end
            trn_step += 1 
        end
        Δpt = (Flux.params(dec) .- initp)
        mse_tst = mean([mse_loss(dec,dtt...) for dtt in data_tst])
        push!(history[:mse_tst],mse_tst)
        push!(history[:ΔP],Δpt)
    end
    return dec, history
end


##
#Linear- Generalized Linear decoders
function ridge_coefficients(V,x;k = 1E-200)
    λ = (V*V' + k*I)\(V*x)
    b = λ'*mean(V,dims=2) .- mean(x)
    return λ, b
end
function mse_linear(V,x; k = 1E-200)
    λ,b = ridge_coefficients(V,x,k=k)
    x_ext = λ'V .+ b
    ε2 = mean((x_ext -x').^2)
    return ε2,λ,b
end
function mse_linear_trntst(data; k = 1E-200)
    data_trn = [data[1]...]
    R_trn = hcat([d[1] for d in data_trn]...)
    x_trn = hcat([d[2] for d in data_trn]...)
    R_tst = hcat([d[1] for d in data[2]]...)
    x_tst = hcat([d[2] for d in data[2]]...)
    λ,b = ridge_coefficients(R_trn,x_trn')
    x_ext = λ'R_tst .+ b
    ε2 = mean((x_ext -x_tst).^2)
    return ε2,λ,b
end
##
function mse_ideal(V,η,x_test,R_t,x_t)
    #Compute losses (cross entropy and mse) on a dataset
    λ_id = V'/η;
    b_id = -sum((V').^2,dims=2)/(2*η)
    H = Flux.softmax(λ_id*R_t .+ b_id)
    x_ext = x_test'*H
    ε2 = mean((x_ext - x_t).^2)
    return ε2,x_ext
end