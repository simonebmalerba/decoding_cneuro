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
    lr = 1f-4              # learning rate
    epochs = 50             # number of epochs
    M = 500                 # latent dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    opt = ADAM             #Optimizer
    f = relu               ##Specify non linearity
    min_diff = 1f-8
end
##
function patience(predicate, wait)
    let count = 0
      function on_trigger(args...; kwargs...)
        count = predicate(args...; kwargs...) ? count + 1 : 0
  
        return count >= wait
      end
    end
end

function plateau(f, width; distance = -, init_score = 0, min_dist = 1f-6)
    is_plateau = let last_score = init_score
      (args...; kwargs...) -> begin
        score = f(args...; kwargs...)
        Δ = abs(distance(last_score, score))
        last_score = score
  
        return Δ < min_dist
      end
    end
  
    return patience(is_plateau, width)
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
    trigger = plateau(last, 5; init_score=0,min_dist=args.min_diff);
    l_av = 0
    progress = Progress(args.epochs)
    for epoch = 1:args.epochs
        for d in data_trn
            l, back = Flux.pullback(ps) do
                mse_loss(dec,d...)
            end
            #Running average
            l_av += l
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            trn_step += 1 
            if trn_step % args.verbose_freq == 0
                push!(history[:mse_trn],l_av/trn_step)
            end
        end
        next!(progress; showvalues=[(:loss, l_av/trn_step),(:epoch,epoch)])
        Δpt = (Flux.params(dec) .- initp)
        mse_tst = mean([mse_loss(dec,dtt...) for dtt in data_tst])
        push!(history[:mse_tst],mse_tst)
        push!(history[:ΔP],Δpt)
        trigger(history[:mse_trn]) && break;
    end
    return dec, history
end



