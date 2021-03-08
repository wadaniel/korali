#! /usr/bin/env python3
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--resultdir', type=str, help='directory to read')

  args = parser.parse_args()
  resultdir = args.resultdir
  print(resultdir)

  resultFiles = [
      f for f in os.listdir(resultdir)
      if os.path.isfile(os.path.join(resultdir, f)) and f.startswith('gen')
  ]
  
  resultFiles = sorted(resultFiles)
  nexperiences = []
  featureweight1 = []
  featureweight2 = []
  featureweight3 = []
  for file in resultFiles:
    with open( resultdir + '/' + file) as f:
      genJs = json.load(f)
      n = genJs['Solver']['Experience Count']
      weights = genJs['Solver']['Feature Weights']
      nexperiences.append(n)
      featureweight1.append(weights[0])
      featureweight2.append(weights[1])
      if len(weights) > 2:
        featureweight3.append(weights[2])

  print(nexperiences)
  plt.plot(nexperiences, featureweight1, label='Weight 1')
  plt.plot(nexperiences, featureweight2, label='Weight 2')
  if len(featureweight3) > 1:
    plt.plot(nexperiences, featureweight3, label='Weight 3')
  plt.title('Feature Weights of Linear Reward Function')
  plt.xlabel('Observation Count')
  plt.ylabel('Weights')
  plt.legend()
  plt.show()
