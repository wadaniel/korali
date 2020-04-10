#!/usr/bin/env python3
import matplotlib.pyplot as plt
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
sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive = 1500)
sampler.run_nested()

res1 = sampler.results

sampler.run_nested(maxcall=10000)
res2 = sampler.results

res1.summary()
res2.summary()


fig, axes = plt.subplots(ndim, 2*ndim+1, figsize=(35, 15))
axes = axes.reshape((ndim, 2*ndim+1))  # reshape axes

# add white space
[a.set_frame_on(False) for a in axes[:, ndim]]
[a.set_xticks([]) for a in axes[:, ndim]]
[a.set_yticks([]) for a in axes[:, ndim]]


# plot initial run (res1; left)
fg, ax = dyplot.cornerplot(res1, color='blue', truths=np.zeros(ndim),
                                truth_color='black', show_titles=True,
                                max_n_ticks=3, quantiles=None,
                                fig=(fig, axes[:, :ndim]))

# plot extended run (res2; right)
fg, ax = dyplot.cornerplot(res2, color='dodgerblue', truths=np.zeros(ndim),
                                truth_color='black', show_titles=True,
                                quantiles=None, max_n_ticks=3,
                                fig=(fig, axes[:, ndim+1:]))
