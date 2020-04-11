#!/usr/bin/env python3

import matplotlib
from matplotlib import pyplot as plt

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

  return -0.5*sse/var - 0.5*len(t)*np.log(2*np.pi*var)

def ptform(u):
    
    return [10. * u[0] - 5., 10. * u[1] - 5, 5 * u[2]]

# "Static" nested sampling.
sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive = 1500, bound = 'single', update_interval = 1.5)
sampler.run_nested()

res1 = sampler.results

sampler.run_nested(maxcall=10000)
res2 = sampler.results


res1.summary()
res2.summary()


fig, axes = plt.subplots(3, 7, figsize=(35, 15))
axes = axes.reshape((3, 7))
[a.set_frame_on(False) for a in axes[:, 3]]
[a.set_xticks([]) for a in axes[:, 3]]
[a.set_yticks([]) for a in axes[:, 3]]


fg, ax = dyplot.cornerplot(res1, color='blue', truths=[0., 0., 0.], truth_color='black',
                           show_titles=True, max_n_ticks=3, title_kwargs={'y': 1.05},
                           quantiles=None, fig=(fig, axes[:, :3]))

# plot extended run (right)
fg, ax = dyplot.cornerplot(res2, color='dodgerblue', truths=[0., 0., 0.], truth_color='black',
                           show_titles=True, title_kwargs={'y': 1.05},
                           quantiles=None, max_n_ticks=3, fig=(fig, axes[:, 4:]))

plt.savefig('myfilename.png')
