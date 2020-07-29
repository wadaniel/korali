import sys
import os
import functools

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, './_model/simple_example_Lavielle'))
sys.path.append('./_model/simple_example_Lavielle')
from model import *
from samplers import *

import numpy as np
import korali


def main():
  # Initialize the distribution
  distrib = SimpleDistributionExponential()

  distrib_S = lambda s: distrib.S(s)
  distrib_zeta = lambda s: distrib.zeta(s)
  distrib_phi = lambda s: distrib.phi(s)

  d3_numberLatentVars = distrib._p.nIndividuals
  d3_numberHyperparams = 1

  d3_initialLatentValues = np.random.normal(0.5, 0.5, (d3_numberLatentVars))
  d3_initialHyperparams = np.random.normal(0, 1, (d3_numberHyperparams))

  sampler = lambda s: distrib.sampleLatent(s)

  k = korali.Engine()
  e = korali.Experiment()

  e["Problem"]["Type"] = "Bayesian/Latent/Exponential"
  e["Problem"]["S Of Likelihood Model"] = distrib_S
  e["Problem"]["Zeta Of Likelihood Model"] = distrib_zeta
  e["Problem"]["Phi Of Likelihood Model"] = distrib_phi
  e["Problem"]["S Dimension"] = distrib.sufficientStatisticsDimension
  e["Problem"]["Latent Variable Sampler"] = sampler

  e["Solver"]["Type"] = "SAEM"
  e["Solver"]["Number Samples Per Step"] = 100
  e["Solver"]["Termination Criteria"]["Max Generations"] = 30

  e["Distributions"][0]["Name"] = "Uniform 0"
  e["Distributions"][0]["Type"] = "Univariate/Uniform"
  e["Distributions"][0]["Minimum"] = -100
  e["Distributions"][0]["Maximum"] = 100

  e["Distributions"][1]["Name"] = "Uniform 1"
  e["Distributions"][1]["Type"] = "Univariate/Uniform"
  e["Distributions"][1]["Minimum"] = -100
  e["Distributions"][1]["Maximum"] = 100

  # * Define which hyperparameters we use (only theta)
  e["Variables"][0]["Name"] = "hyperparameter"
  e["Variables"][0]["Bayesian Type"] = "Hyperparameter"
  e["Variables"][0][
      "Prior Distribution"] = "Uniform 1"  # not used (?) but required
  e["Variables"][0]["Initial Value"] = d3_initialHyperparams[0]
  e["Variables"][0]["Upper Bound"] = e["Distributions"][1]["Maximum"]
  e["Variables"][0]["Lower Bound"] = e["Distributions"][1]["Minimum"]

  # * Latent variables
  latent_counter = 0
  for i in range(distrib._p.nIndividuals):
    e["Variables"][i + 1]["Name"] = "individual_mean_" + str(latent_counter)
    e["Variables"][i + 1]["Bayesian Type"] = "Latent"
    e["Variables"][i + 1]["Prior Distribution"] = "Uniform 0"
    e["Variables"][i +
                   1]["Initial Value"] = d3_initialLatentValues[latent_counter]
    latent_counter += 1

  e["File Output"]["Frequency"] = 1
  e["Console Output"]["Frequency"] = 10
  e["Console Output"]["Verbosity"] = "Detailed"

  k.run(e)

  print("------------ Experiment finished ------------\n")
  print("   Compare results to true optimizer in ")
  print("     '_data/simple_example_Lavielle/data_925_info.txt' ")
  print("   Plot experiment stats with: ")
  print("     'python3 -m korali.plotter'")


if __name__ == '__main__':
  # ** For debugging, try this: **
  # import sys, trace
  # sys.stdout = sys.stderr
  # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
  # tracer.runfunc(main)
  # ** Else: **
  main()
