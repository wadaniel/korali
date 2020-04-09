#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import dynesty
from dynesty import plotting as dyplot

ndim = 3

def loglike(x): 
  v1 = x[0]
  v2 = x[1]
  var = x[2]*x[2]

  t   = [1.0, 2.0, 3.0, 4.0, 5.0]
  ref = [3.2069, 4.1454, 4.9393, 6.0588, 6.8425]
  
  sse = 0.0
  for i in range(len(t)):
    sse += (v1*t[i] + v2 - ref[i])**2

  return -0.5*sse/var - 0.5*np.log(2*np.pi*var)

def ptform(u):
    
    return [10. * u[0] - 5., 10. * u[1] - 5, 5 * u[2]]

# "Static" nested sampling.
sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive = 1500)
sampler.run_nested()

res1 = sampler.results

sampler.run_nested(maxcall=10000)
res2 = sampler.results

res1.summary()
res2.summary()


fig, axes = plt.subplots(3, 7, figsize=(35, 15))
axes = axes.reshape((3, 7))  # reshape axes

# add white space
[a.set_frame_on(False) for a in axes[:, 3]]
[a.set_xticks([]) for a in axes[:, 3]]
[a.set_yticks([]) for a in axes[:, 3]]


# plot initial run (res1; left)
fg, ax = dyplot.cornerplot(res1, color='blue', truths=np.zeros(ndim),
                                truth_color='black', show_titles=True,
                                max_n_ticks=3, quantiles=None,
                                fig=(fig, axes[:, :3]))

# plot extended run (res2; right)
fg, ax = dyplot.cornerplot(res2, color='dodgerblue', truths=np.zeros(ndim),
                                truth_color='black', show_titles=True,
                                quantiles=None, max_n_ticks=3,
                                fig=(fig, axes[:, 4:]))
