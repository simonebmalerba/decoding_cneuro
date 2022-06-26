using DrWatson
@quickactivate "decoding_cneuro"
using Distributions, StatsBase , LinearAlgebra, MultivariateStats,Roots
using KernelFunctions

function rational_quadratic_normalizer(σVec,αVec,x_trn;N=500)
    #Compute normalization constant for the SNR in the case
    #of rational quadratic kernel
    A1matrix = zeros(length(σVec),length(αVec))
    AVec = 1:1:15
    for (j,αk) in enumerate(αVec)
        k = RationalQuadraticKernel(α = αk)
        for (i,σi) = enumerate(σVec)
            t = ScaleTransform(1/(sqrt(2)*σi))
            Vs = []
            for A = AVec
                K = kernelmatrix(A*k,t(x_trn))
                R_ex = mvn_sample(K,N)
                push!(Vs,mean(var(R_ex,dims=2)))
            end
            m = cov(Vs,AVec)/var(AVec)
            b = mean(Vs) - m*mean(AVec)
            A1 = (1-b)/m
            K = kernelmatrix(A1*k,t(x_trn))
            R_ex = mvn_sample(K,N)
            @info σi,αk,A1,mean(var(R_ex,dims=2))
            A1matrix[i,j] = A1
        end
    end
    return A1matrix
end
#Gaussian Processes
function mvn_sample(K,N;λ = 1f-10)
    #Sample N realizations from a gaussian process with covariance matrix K
    num_inputs,_ = size(K)
    L = cholesky(K + λ* I)
    v = randn(Float32,num_inputs, N)
    f = L.L * v
    return f'
end

function conditional_sampling(R_trn,K_trn,k_trntst,K_tst,nsamples; λ = 1f-5)
    #Sample nsamples realizations from the marginal distribution of the new
    #Neural responses given the old ones in R_trn
    N,_ = size(R_trn)
    ntst,_ = size(K_tst)
    #Given a set of new points, and old responses, compute the next responses
    m = R_trn*(inv(K_trn)*k_trntst)
    L = cholesky(Hermitian(K_tst - k_trntst'inv(K_trn+λ*I)*k_trntst) +λ*I)
    f = []
    for i=1:nsamples
        v = randn(Float32,ntst,N)
        push!(f,(L.L * v)' + m)
    end
    return f
end

#Effective regularize by Jacot et al. (2020)
function effective_ridge(λ,d,N)
    f(λ_e) = λ_e - λ - λ_e/N*sum(d ./(d.+λ_e))
    λ_s = find_zero(f,λ+ 0.05)
    dλ_s = 1/(1-(1/N)*sum((d ./(d.+λ_s)).^2))
    return λ_s,dλ_s
end
function effective_ridge0(d,N)
    f(λ_e) = 1  - 1/N*sum(d ./(d.+λ_e))
    λ_s = find_zero(f,0.01)
    dλ_s = 1/(1-(1/N)*sum((d ./(d.+λ_s)).^2))
    return λ_s,dλ_s
end