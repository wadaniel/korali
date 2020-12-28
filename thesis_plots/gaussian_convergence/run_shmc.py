import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#!/usr/bin/env python3
import korali
import numpy as np
from scipy.stats import t
from models import * 

distr = "Gaussian"
mu = 0.0
sigma = 1.0
y_max = 1.0
x_range = 1.0
if distr == "Gaussian":
    model = lambda x: lGaussianModel(x, mu, sigma)
    pdf = lambda x: gaussianDensity(x, mu, sigma)
    x_range = 5.0 * sigma  
    y_max = 0.5 / sigma
elif distr == "Laplace":
    model = lambda x: lLaplaceModel(x, mu, sigma)
    pdf = lambda x: laplaceDensity(x, mu, sigma)
    x_range = 5.0 * sigma
    y_max = 0.8 / sigma
elif distr == "Uniform":
    model = lambda x: lUniformModel(x, mu, sigma)
    pdf = lambda x: uniformDensity(x, mu, sigma) 
    x_range = 0.8 * np.ceil(10 * ((np.sqrt(12) * sigma) + 0.05)) / 10.0
    y_max = np.ceil(10 * (1.0 / (np.sqrt(12) * sigma) + 0.05)) / 10.0

dim = 1
numSamples = 10000
numExperiments = 1
samples = np.zeros(shape=(numExperiments, numSamples))
for experiment in range(numExperiments):
    # Set Random Seed
    e = korali.Experiment()
    e["Random Seed"] = 0xC0FFEE + experiment
    #######################################################################################
    ############################# File/Console output settings ############################
    #######################################################################################
    e["File Output"]["Frequency"] = 1000
    e["Console Output"]["Frequency"] = 1000
    # e["Console Output"]["Verbosity"] = "Detailed"

    #######################################################################################
    ################################ Variable Initilization ###############################
    #######################################################################################
    e["Variables"][0]["Name"] = "X"
    e["Variables"][0]["Initial Mean"] = (2.0 * np.random.rand() - 1.0) * sigma
    e["Variables"][0]["Initial Standard Deviation"] = 1.0

    #######################################################################################
    ################################## Sampler version ####################################
    #######################################################################################
    e["Problem"]["Type"] = "Sampling"
    e["Solver"]["Type"] = "Sampler/HMC"
    e["Solver"]["Version"] = 'Static'
    e["Solver"]["Use Diagonal Metric"] = False #for Version != Static || Version != Riemannian
    e["Solver"]["Inverse Regularization Parameter"] = 1.0 #for Version = Riemannian || Version == Riemannian Const
    e["Problem"]["Probability Function"] = model

    #######################################################################################
    ############################# Number of samples settings ##############################
    #######################################################################################
    e["Solver"]["Burn In"] = 1000
    e["Solver"]["Termination Criteria"]["Max Samples"] = numSamples
    e["Solver"]["Initial Fast Adaption Interval"] = 75
    e["Solver"]["Final Fast Adaption Interval"] = 50
    e["Solver"]["Initial Slow Adaption Interval"] = 25

    #######################################################################################
    ############################# Integration length settings #############################
    #######################################################################################
    # General Step Size settings
    e["Solver"]["Step Size"] = 0.10
    e["Solver"]["Step Size Jitter"] = 0.0
    e["Solver"]["Num Integration Steps"] = 3
    # Adaptive Step Size settings
    e["Solver"]["Use Adaptive Step Size"] = False
    e["Solver"]["Target Acceptance Rate"] = 0.7 #for Use Adaptive Step Size = True
    e["Solver"]["Target Integration Time"] = 1.0 #for Use Adaptive Step Size = True
    e["Solver"]["Adaptive Step Size Speed Constant"] = 0.05 #for Use Adaptive Step Size = True
    e["Solver"]["Adaptive Step Size Stabilization Constant"] = 10.0 #for Use Adaptive Step Size = True
    e["Solver"]["Adaptive Step Size Schedule Constant"] = 0.75 #for Use Adaptive Step Size = True
    # NUTS settings
    e["Solver"]["Use NUTS"] = True
    e["Solver"]["Max Depth"] = 4 #for Use NUTS = True
    # Implicit leapfrog settings
    e["Solver"]["Max Integration Steps"] = 100 #for Version = Riemannian
    
    #######################################################################################
    ################################### Run Experiment ####################################
    #######################################################################################
    k = korali.Engine()
    k.run(e)
    experimentSamples = e["Results"]["Sample Database"]
    experimentSamples = np.reshape(experimentSamples, (-1, dim)).flatten()
    samples[experiment, :] = experimentSamples


#######################################################################################
###################################### Plotting #######################################
#######################################################################################

# Plot a single histogram #############################################################
fig, ax = plt.subplots()
ax.set_axisbelow(True)

suffixStr = distr+"_experiments_"+str(numExperiments)+"_samples_"+str(numSamples)
plt.xlim(mu-x_range, mu+x_range)
plt.ylim(0.0, y_max)
ax.hist(samples[0, :], bins=50, density=True, zorder=1)

numPoints = 1000
xVals = np.linspace(-5.0 * sigma, 5.0 * sigma, numPoints)
stdNormalDistrDensity = np.array([pdf(x) for x in xVals])

plt.xlabel("Position $q$", fontsize=16)
plt.ylabel("Rel. Occurrence", fontsize=16)
plt.tick_params(axis='both', labelsize=14)
ax.plot(xVals, stdNormalDistrDensity, 'r-')


plt.grid(linestyle=":", zorder=-1)
plt.legend(["PDF", "Samples"], prop={"size": 14})
plt.savefig("./shmc/samples_"+suffixStr+".svg", bbox_inches="tight")
plt.savefig("./shmc/samples_"+suffixStr, dpi=200, bbox_inches="tight")

plt.clf()

# Plot Convergence ####################################################################
cumSumSamples = np.cumsum(samples, axis=1)
print("cumSumSamples = ", cumSumSamples)
numSamplesRange = np.arange(1, numSamples+1)
empiricalExpectation = np.zeros(shape=(numExperiments, numSamples))
for experiment in range(numExperiments):
    empiricalExpectation[experiment, :] = cumSumSamples[experiment, :] / numSamplesRange
print("empiricalExpectation = ", empiricalExpectation)

errorExpectation = np.abs(mu - empiricalExpectation)
averageErrorExpectation = np.mean(errorExpectation, axis=0)
print("errorExpectation = ", errorExpectation)
errorIdeal = np.array([1.0/np.sqrt(n) for n in numSamplesRange])

# Calculate confidence interval
alpha = 0.05
confidenceValue = 1.0 - alpha / 2.0
studentTCoeff = t.ppf(confidenceValue, df=numExperiments-1)
ci = studentTCoeff * np.std(errorExpectation, axis=0) / np.sqrt(numExperiments)

fig, ax = plt.subplots()
plt.xlim(numSamplesRange[0], numSamplesRange[-1])
plt.ylim(1.0/5.0 * errorIdeal[-1], 5*errorIdeal[0])
ax.loglog(numSamplesRange, errorExpectation[0, :], 'silver', linewidth=0.3, zorder=-1)
ax.loglog(numSamplesRange, errorIdeal, 'g--')
# ax.loglog(numSamplesRange, LSQFit, 'r--')
ax.loglog(numSamplesRange, averageErrorExpectation)
print("averageErrorExpectation = ", averageErrorExpectation)
ax.fill_between(numSamplesRange, (averageErrorExpectation-ci), (averageErrorExpectation+ci), color='b', alpha=.1)
ax.tick_params(axis='both', labelsize=14)
plt.grid(linestyle=":", zorder=-2)
plt.xlabel("Number of Samples $N$", fontsize=16)
plt.ylabel("Error $|\\mathbb{E}[Q]-\hat{E}[Q]|$", fontsize=16)
plt.legend(["Single Run", "$N^{-\\frac{1}{2}}$", "$\hat{E}[Error]$" ,str((100 * (1.0 - alpha))) + "% CI"], loc="lower left", prop={"size": 14})
plt.savefig("./shmc/convergence_"+suffixStr+".svg", bbox_inches="tight")
plt.savefig("./shmc/convergence_"+suffixStr+".png", dpi=200, bbox_inches="tight")