using DrWatson
@quickactivate "decoding_cneuro"
using Flux: softmax
## Utility functions
function mse_ideal(V,η,x_m,R,x_tst)
    #Ideal mean squared error for neural responses R to a test
    #set of stimuli given the mean
    #tuning curves V
    λ_id = V'/η;
    b_id = -sum((V').^2,dims=2)/(2*η)
    H = softmax(λ_id*R .+ b_id)
    x_ext = x_m'*H
    ε = mean((x_ext - x_tst).^2)
    return ε
end
function participation_ratio(X)
    C = cov(X)
    μ = eigvals(C)
    PR = sum(μ)^2/sum(μ.^2)
    return PR
end