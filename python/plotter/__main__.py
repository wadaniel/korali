#! /usr/bin/env python3
import os
import sys
import json
import argparse
from korali.plotter.cmaes import plot_cmaes
from korali.plotter.tmcmc import plot_tmcmc
from korali.plotter.mcmc import plot_mcmc

def main(live):
 path = '_korali_result'
 firstResult = path + '/s00000.json'
 if ( not os.path.isfile(firstResult) ):
  print("[Korali] Error: Did not find any results in the _korali_result folder...")
  exit(-1)

 with open(firstResult) as f:
  data  = json.load(f)
  method = 'bla'

 if 'TMCMC' in data['Solver']:
  print("[Korali] Running TMCMC Plotter...")
  plot_tmcmc(path, live)
  exit(0)
 
 if 'MCMC' in data['Solver']:
  print("[Korali] Running MCMC Plotter...")
  plot_mcmc(path, live)
  exit(0)

 if ( data['Solver']['CMA-ES'] or data['Solver']['CCMA-ES'] ):
  print("[Korali] Running CMA-ES Plotter...")
  plot_cmaes(path, live)
  exit(0)

 if ( data['Solver']['DE'] ):
  print("[Korali] Error: DE plotting not yet implemented...")
  exit(-1)

 print("[Korali] Error: Did not recognize method for plotting...")
 exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='korali.plotter', description='Process korali results in _korali_results folder.')
    parser.add_argument('--live', help='run live plotting', action='store_true')
    args = parser.parse_args()
    
    main(args.live)
