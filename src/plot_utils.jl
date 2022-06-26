using DrWatson
@quickactivate "decoding_cneuro"
using Plots
using LaTeXStrings
#Create array of colors
C(g::ColorGradient,n) = RGB[g[z] 
    for z=Int.(round.((range(1,stop=256,length=n))))]

f1 = Plots.font("sans-serif",12)
f2 = Plots.font("sans-serif",16)
fs = Dict(:guidefont => f2, :legendfont => f1, :tickfont => f1)
default(; fs...)