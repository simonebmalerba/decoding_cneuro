# Decoding complex neural responses
## Introduction
The project aims to analyze the ability of different machine learning decoding methods in extracting information from noisy neural responses.
We use the model in Random Compressed Coding with Neurons, Blanco Malerba S, et al., bioRxiv 2022.01.06.475186; doi: https://doi.org/10.1101/2022.01.06.475186
to generate neural responses: the mean response of a neuron, i, as a function of stimulus, x, is sampled from a Gaussian process with a Gaussian kernel of bandwidth $\sigma$, v_i(x).
Neural responses are affected by i.i.d. Gaussian noise. 

We test different methods to decode the value of the stimulus, x, from the noisy response pattern of a population of N neurons, and we analyze the decoding accuracy, or mean squared error, as a function of the parameters of the encoder, population size and tuning width, and of the decoder, number of decoding neurons and training regime.
- ideal decoding method, or Bayesian posterior mean (utils.jl)
- probabilistic decoder, trained with cross entropy loss to maximize the probability of the correct label (probabilistic_decoder.jl)
- MLP with one hidden layer of size M (dnn_decoder.jl)
- Neural Tangent Kernel: kernel machine corresponding to the infinite width regime of a MLP, trained in the lazy regime (large variance of the weights at initialization) (NTK)
