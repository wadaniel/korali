#! /usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from korali.plotter.helpers import hlsColors, drawMulticoloredLine
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

plotSamples = True

#Plot scatter plot in upper triangle of figure
def plot_upper_triangle(ax, theta, f=None):
  dim = theta.shape[1]
  for i in range(dim):
    for j in range(i + 1, dim):
      if f:
        ax[i, j].scatter(
            theta[:, i], theta[:, j], marker='o', s=3, alpha=0.5, c=f)
      else:
        ax[i, j].plot(theta[:, i], theta[:, j], '.', markersize=3)
      ax[i, j].grid(b=True, which='both')
      ax[i, j].set_xlabel("F"+str(i))
      ax[i, j].set_ylabel("F"+str(j))


#Plot scatter plot in lower triangle of figure
def plot_lower_triangle(ax, theta, f=None):
  dim = theta.shape[1]
  for i in range(dim):
    for j in range(0, i):
      if f:
        ax[i, j].scatter(
            theta[:, i], theta[:, j], marker='o', s=3, alpha=0.5, c=f)
      else:
        ax[i, j].plot(theta[:, i], theta[:, j], '.', markersize=3)
      ax[i, j].grid(b=True, which='both')
      ax[i, j].set_xlabel("F"+str(i))
      ax[i, j].set_ylabel("F"+str(j))


def plotGen(genList, idx):
  numgens = len(genList)

  lastGen = 0
  for i in genList:
    if genList[i]['Current Generation'] > lastGen:
      lastGen = genList[i]['Current Generation']

  numdim = genList[lastGen]['Solver']['Num Objectives']
  sampleVals = np.array(genList[lastGen]['Solver']['Sample Value Collection'])

  isFinite = [~np.isnan(s - s).any() for s in sampleVals]  # Filter trick
  sampleVals = sampleVals[isFinite]

  numentries = len(sampleVals)

  fig, ax = plt.subplots(numdim, numdim, figsize=(8, 8))
  samplesTmp = np.reshape(sampleVals, (numentries, numdim))
  plt.suptitle(
      'MO-CMA-ES Plotter - \nNumber of Samples {0}'.format(str(numentries)),
      fontweight='bold',
      fontsize=12)

  if plotSamples and numdim > 1:
      plot_upper_triangle(ax, samplesTmp)
      plot_lower_triangle(ax, samplesTmp)
    
      for i in range(numdim):
        ax[i, i].set_xticklabels([])
        ax[i, i].set_yticklabels([])


def plot(genList, args, addons):
  print(addons)
  numgens = len(genList)

  plotAll = args.all
  if plotAll:
    for idx in genList:
      plotGen(genList, idx)
  else:
    lastGen = 0
    for i in genList:
      if genList[i]['Current Generation'] > lastGen:
        lastGen = genList[i]['Current Generation']
    plotGen(genList, lastGen)
