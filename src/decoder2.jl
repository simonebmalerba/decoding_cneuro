using DrWatson
@quickactivate "decoding_cneuro"
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



