#!/usr/bin/env python3

import csv
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
import korali

import sys
sys.path.append('../_optimization_model/_rl_model')
from evalenv import *


def readActions(obsfile):

 with open(obsfile) as csv_file:
    obsactions = []
    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        obsactions.append(row[4])

    return obsactions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--obsfiles', type=str, nargs='+', help='Observation file to read', required=True)
    parser.add_argument('--policies', type=str, nargs='+', help='Observation file to read', required=True)

    args = parser.parse_args()

    files = args.obsfiles
    policies = args.policies

    perrors = []
    for policy in policies:
      
      oerrors = []
      for infile in files:
  
        k = korali.Engine()
        e = korali.Experiment()
   
        policyfile = policy + '/latest'
        found = e.loadState(policyfile)
        if (found == False):
            print('Previous run {} not found, exit...'.format(policyfile))
            sys.exit()
 
        e["Problem"]["Environment Function"] = evalenv
        e["Problem"]["Custom Settings"]["Input"] = infile
        e["Solver"]["Mode"] = "Testing"
        e["Solver"]["Testing"]["Sample Ids"] = [0]
        e["File Output"]["Enabled"] = False
        
        k.run(e)

        suml2error = e["Solver"]["Testing"]["Reward"][0]
        oerrors.append(suml2error)

      perrors.append(oerrors)

    pmuerrors = []
    pminerrors = []
    psdeverrors = []
    for errors in perrors:
      minimum = min(errors)
      mu = np.mean(errors)
      sdev = np.std(errors)
      pminerrors.append(minimum)
      pmuerrors.append(mu)
      psdeverrors.append(sdev)

    adjust = np.ones(len(pmuerrors)) * 0.5
    adjust[-1] = 0.

    print(pminerrors)
    x = np.power(0.5,range(len(pmuerrors))) * adjust
    plt.errorbar(x=x, y=pmuerrors, yerr=psdeverrors)
    plt.tight_layout()
    plt.savefig('policies.png')






