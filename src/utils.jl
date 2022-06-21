using DrWatson
@quickactivate "Random_coding"
## Utility functions
function participation_ratio(X)
    C = cov(X)
    μ = eigvals(C)
    PR = sum(μ)^2/sum(μ.^2)
    return PR
end