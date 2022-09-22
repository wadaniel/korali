#! /usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--resultdir','--d', required=True, type=str, help='directory to read')
  parser.add_argument('--logy', action='store_true', help='logscale y axis')
  parser.add_argument('--intermediate', '--i', action='store_true', help='show intermediate plots')
  parser.add_argument('--last', '--l', action='store_true', help='show last estimate')
  parser.add_argument('--fusion', '--f', action='store_true', help='plot fusion distribution')
  parser.add_argument('--both', '--b', action='store_true', help='plot single and fusion distribution')
  parser.add_argument('--returns', '--r', action='store_true', help='plot returns')

  args = parser.parse_args()
  resultdir = args.resultdir
  showIntermediate = args.intermediate
  showOnlyLast = args.last
  plotFusion = args.fusion
  plotBoth = args.both
  plotReturns = args.returns

  print(resultdir)

  resultFile = "{}/latest".format(resultdir)
  
  nexp = None
  batchsize = None
  stats = []
  statsFusion = []
  returns = []
  thetas = None
  with open( resultFile ) as f:
      genJs = json.load(f)
      nexp = genJs['Solver']['Experiences Between Partition Function Statistics']
      batchsize = genJs['Solver']['Background Batch Size']
      thetas = genJs['Solver']['Statistic Feature Weights']
      stats = genJs['Solver']['Statistic Log Partition Function']
      statsFusion = genJs['Solver']['Statistic Fusion Log Partition Function']
      returns = genJs['Solver']['Statistic Cumulative Rewards']

  labelRewards = "Trajectory Returns"
  for i, means in enumerate(stats):
    bsize = len(means)
    xbatchsize = range(1,len(means)+1)
    if showIntermediate:
        label = "Single"
        labelFusion = "Mixture"
 
    else:
        label = r"Total Trajectories {} ($\theta$ = {:.3f}, Single)".format(bsize, thetas[i][0])
        labelFusion = r"Total Trajectories {} ($\theta$ = {:.3f}, Mixture)".format(bsize, thetas[i][0])
 
    if showOnlyLast == True and i < len(stats)-1:
        continue

    if plotBoth:
        plt.plot(xbatchsize, means, label=label, linestyle='-')
        plt.plot(xbatchsize, statsFusion[i], label=labelFusion, linestyle='--')
    else:
        if plotFusion:
            plt.plot(xbatchsize, statsFusion[i], label=labelFusion, linestyle='--')
        else:
            plt.plot(xbatchsize, means, label=label, linestyle='-')
 
    plt.xlabel('Number of Trajectories M')
    plt.ylabel(r'log($Z_\theta$)')
    if(showIntermediate or showOnlyLast == True):
        plt.title(r'Estimate Log Partition Function $Z_\theta$ ($\theta = {:.3f}$, Total Trajectories {})'.format(thetas[i][0], bsize))
        if plotReturns:
            plt.plot(xbatchsize, returns[i], label=labelRewards, linestyle='None', marker='x', color='r', alpha=0.5)
        plt.legend()
        plt.show()
 
  if(not (showIntermediate == True or showOnlyLast == True)):
    plt.title(r'Estimate Log Partition Function $Z_\theta$')
    if plotReturns:
        plt.plot(xbatchsize, returns[-1], label=labelRewards, linestyle='None', marker='x', color='r', alpha=0.5)
    plt.legend()
    plt.show()
    
