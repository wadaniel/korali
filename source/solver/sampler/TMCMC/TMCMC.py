#! /usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from korali.plotter.helpers import hlsColors, drawMulticoloredLine
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sampler import plot_histogram, plot_upper_triangle, plot_lower_triangle

def plot(js):
    numdim = len(js['Variables'])
    samples = js['Solvers'][-1]['Internal']['Sample Database']
    numentries = len(samples)
    
    fig, ax = plt.subplots(numdim, numdim, figsize=(8,8))
    samplesTmp = np.reshape( samples, (numentries,numdim) )
    plt.suptitle( 'TMCMC Plotter - \nNumber of Samples {0}\n'.format(str(numentries)), fontweight='bold', fontsize  = 12)
    plot_histogram(ax, samplesTmp)
    plot_upper_triangle(ax, samplesTmp, False)
    plot_lower_triangle(ax, samplesTmp)
    plt.show()
