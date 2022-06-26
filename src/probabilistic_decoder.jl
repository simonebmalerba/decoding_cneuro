using DrWatson
@quickactivate "decoding_cneuro"
using Flux
using Parameters: @with_kw
using ProgressMeter: Progress, next!

@with_kw mutable struct TrainArgsProb
    lr = 1e-3              # learning rate
    epochs = 20             # number of epochs
    M = 500                 # latent dimension
    verbose_freq = 20       # logging for every verbose_freq iterations
    opt = ADAM             #Optimizer
    min_diff = 1e-2 # Minimum difference for stopping criterium
end

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

function one_hotdataset(x_trn,x_tst,R_trn,R_tst,M;bs=50)
    #Convert real data to one-hot vectors
    # M: number of bins. Return data and bin centers
    P = length(x_trn)
    bin = range(-0.5,0.5,length=M+1)
    h = fit(Histogram,vcat(x_trn,x_tst),bin)
    x_map = StatsBase.binindex.(Ref(h), vcat(x_trn,x_tst))
    x_m = bin[1:end-1] .+ diff(bin)/2
    x_10 = Flux.onehotbatch(x_map,1:M);
    x_10trn = x_10[:,1:P]; x_10tst = x_10[:,P+1:end];
    data_10trn = Flux.Data.DataLoader((Float32.(R_trn),x_10trn),
        batchsize = bs,shuffle = true);
    data_10tst = Flux.Data.DataLoader((Float32.(R_tst),x_10tst),
        batchsize = 100,shuffle = false);
    return [data_10trn,data_10tst],x_m
end

##Loss function
x_entropy(dec,r,x_10) = Flux.Losses.crossentropy(dec(r),x_10)
## MSE
function mse_prob(dec,data,x_tst,x_m)
    x_ext = vec(hcat([x_m'dec(d[1]) for d in data]...))
    return Flux.mse(x_ext,x_tst)
end
#
##
function train_prob_decoder(data,x_m,x_tst; dec=nothing,kws...)
    # Train a probabilistic decoder on the data generated by the RFFN:
    # data= [data_trn, data_tst], where x_trn as one-hot vectors
    # x_m: preferrred positions of decoder neurons
    # x_tst: [ntst,1]: test data as real numbers
    data_trn,data_tst = data[1],data[2]
    N,mb = size(first(data_trn)[1])
    args = TrainArgsProb(; kws...)
    if isnothing(dec)
        #Decoder is a linear network which output a probability
        # distribution over M values
        dec = Chain(Dense(N,args.M),softmax)   
    end
    #Optmizer for gradient descent
    opt = args.opt(args.lr)          
    #Collect parameters             
    ps = Flux.params(dec);  
    #Dictionary to trace progress of training  
    history = Dict(:ce_trn => Float32[],:ce_tst => Float32[],
        :mse_tst =>Float32[])     
    trn_step =0
    #trigger for plateau of training error: difference from last ce_trn
    #constantly below 1e-3 (in 5 epochs)
    trigger = plateau(last, 5; init_score=0,min_dist=args.min_diff);
    l_av = 0
    progress = Progress(args.epochs)
    for epoch = 1:args.epochs
        #@info "Epoch $(epoch)"
        for d in data_trn
            l, back = Flux.pullback(ps) do
                x_entropy(dec,d...) 
            end
            #Running average
            l_av += l 
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            trn_step += 1
            if trn_step % args.verbose_freq == 0
                push!(history[:ce_trn],l_av/trn_step)
            end
        end
        next!(progress; showvalues=[(:loss, l_av/trn_step),(:epoch,epoch)])
        push!(history[:ce_tst],
            mean([x_entropy(dec,dtt...) for dtt in data_tst]))
        push!(history[:mse_tst],mse_prob(dec,data_tst,x_tst,x_m))
        trigger(history[:ce_trn]) && break;
        #print(history[:ce_trn][end]-history[:ce_trn][end-ie])
    end
    return dec,history
end



