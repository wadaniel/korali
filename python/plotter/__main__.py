#! /usr/bin/env python3
import os
import sys
import json
import argparse
from korali.plotter.cmaes import plot_cmaes
from korali.plotter.tmcmc import plot_tmcmc

def main(live):
 path = '_korali_result'
 firstResult = path + '/s00000.json'
 if ( not os.path.isfile(firstResult) ):
  print("[Korali] Error: Did not find any results in the _korali_result folder...")
  exit(-1)

 with open(firstResult) as f:
  data  = json.load(f)
  method = data['Solver']['Method']

 if ( method == "TMCMC" ):
  print("[Korali] Running TMCMC Plotter...")
  plot_tmcmc(path, live)
  exit(0)

 if ( method == "CMA-ES" or method == "CCMA-ES" ):
  print("[Korali] Running CMA-ES Plotter...")
  plot_cmaes(path, live)
  exit(0)

 print("[Korali] Error: Did not recognize method for plotting...")
 exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='korali.plotter', description='Process korali results in _korali_results folder.')
    parser.add_argument('--live', help='run live plotting', action='store_true')
    args = parser.parse_args()
    
    main(args.live)
