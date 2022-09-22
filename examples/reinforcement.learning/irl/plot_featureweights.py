#! /usr/bin/env python3
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--resultdir', type=str, help='directory to read')
  parser.add_argument('--logy', action='store_true', help='logscale y axis')

  args = parser.parse_args()
  resultdir = args.resultdir
  print(resultdir)

  resultFiles = [
      f for f in os.listdir(resultdir)
      if os.path.isfile(os.path.join(resultdir, f)) and f.startswith('gen')
  ]
  
  resultFiles = sorted(resultFiles)
  nexperiences = []
  featureweights = []
  for file in resultFiles:
    with open( resultdir + '/' + file) as f:
      genJs = json.load(f)
      n = genJs['Solver']['Experience Count']
      weights = genJs['Solver']['Feature Weights']
      if (args.logy):
          weights = [ abs(w) for w in weights ] 
      nexperiences.append(n)
      featureweights.append(weights)

  plt.plot(nexperiences, featureweights)
  #plt.title('Feature Weights of Linear Reward Function')
  plt.title('Target Angle in Parametrized Reward Function')
  plt.xlabel('Observation Count')
  #plt.ylabel('Weights')
  plt.ylabel('Absolute Target Angle')
  if(args.logy):
      plt.yscale('log')
  plt.legend()
  plt.show()
