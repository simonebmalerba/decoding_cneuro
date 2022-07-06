using DrWatson
@quickactivate "decoding_cneuro"
#using TensorBoardLogger,Logging
using Plots
using StatsBase,Distributions
include(srcdir("plot_utils.jl"))
##
mydir= datadir("sims/dnn_decoder")
flnames = filter(x ->occursin(r"dnn_dec",x),
    readdir(mydir))
myf = flnames[2]
data = load(datadir(mydir,myf))