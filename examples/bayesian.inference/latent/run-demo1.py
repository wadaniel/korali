#!/usr/bin/env python3

import sys
sys.path.append('./_model')
import numpy as np

from demo1 import *
import korali

def main():

    # the generative model is a Gaussian mixture
    gen_model = generative_model()
    data = gen_model.generate_data(10)

    m = model(data)

    k = korali.Engine()
    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Latent/Exponential"
    e["Problem"]["S Of Likelihood Model"] = m.S
    e["Problem"]["Zeta Of Likelihood Model"] = m.zeta
    e["Problem"]["Phi Of Likelihood Model"] = m.phi
    e["Problem"]["S Dimension"] = 1
    e["Problem"]["Latent Variable Sampler"] = lambda sample: m.sample_latent(sample)

    e["Solver"]["Type"] = "SAEM"
    e["Solver"]["Number Samples Per Step"] = 10
    e["Solver"]["Termination Criteria"]["Max Generations"] = 2000

    e["Variables"][0]["Name"] = "sigma_square"
    e["Variables"][0]["Bayesian Type"] = "Hyperparameter"
    e["Variables"][0]["Prior Distribution"] = "Uniform 0"
    e["Variables"][0]["Initial Value"] = 5.0
    e["Variables"][0]["Upper Bound"] = 15
    e["Variables"][0]["Lower Bound"] = 0

    # define a variable for each coordinate of mu
    for i in range(m.dim):
        e["Variables"][1 + i]["Name"] = "mu" + str(i)
        e["Variables"][1 + i]["Bayesian Type"] = "Latent"
        e["Variables"][1 + i]["Prior Distribution"] = "Uniform 1"
        e["Variables"][1 + i]["Initial Value"] = 0

    e["Distributions"][0]["Name"] = "Uniform 0"
    e["Distributions"][0]["Type"] = "Univariate/Uniform"
    e["Distributions"][0]["Minimum"] = 0
    e["Distributions"][0]["Maximum"] = 5

    e["Distributions"][1]["Name"] = "Uniform 1"
    e["Distributions"][1]["Type"] = "Univariate/Uniform"
    e["Distributions"][1]["Minimum"] = -5
    e["Distributions"][1]["Maximum"] = 5

    e["File Output"]["Path"] = '_korali_result_saem'
    e["Random Seed"] = 0xC0FFEE

    k.run(e)


if __name__ == '__main__':
    main()
