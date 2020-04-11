#!/usr/bin/env python3
import matplotlib
from matplotlib import pyplot as plt

import numpy as np
import dynesty
from dynesty import plotting as dyplot


ndim = 1

def loglike(x): 
  v1 = x[0]
  return -0.5*v1*v1 - 0.5*np.log(2*np.pi)

def ptform(u):
    
    return [10. * u[0] - 5.]

# "Static" nested sampling.
sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive = 1500, niter = 10, add_live = True)
sampler.run_nested()

res1 = sampler.results

res1.summary()

print(res1.samples[1:10])
print(res1.logwt[1:10])
print(res1.logvol[1:10])

plt.savefig('custom.png')
