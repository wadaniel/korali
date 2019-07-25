#! /usr/bin/env python3
import os
import sys
import signal
import json
import argparse
import matplotlib

def main(path, allFiles, live, generation, mean, check, test):

 if (check == True):
  print("[Korali] Plotter correctly installed.")
  exit(0)
 
 if (test == True):
     matplotlib.use('Agg')

 if ( (allFiles == True) and (generation is not None)):
    print("korali.plotter: error: argument --all and argument --generation "\
            "GENERATION cannot be combined")
    exit(-1)

 from korali.plotter.helpers import sig
 from korali.plotter.cmaes import plot_cmaes
 from korali.plotter.tmcmc import plot_tmcmc
 from korali.plotter.mcmc import plot_mcmc
 from korali.plotter.dea import plot_dea

 signal.signal(signal.SIGINT, sig)
 
 firstResult = path + '/s00000.json'
 if ( not os.path.isfile(firstResult) ):
  print("[Korali] Error: Did not find any results in the {0} folder...".format(path))
  exit(-1)

 with open(firstResult) as f:
  data  = json.load(f)
 
 solver = data['Solver']['Type']
 if ( 'TMCMC' == solver ):
   print("[Korali] Running TMCMC Plotter...")
   plot_tmcmc(path, allFiles, live, generation, test)
   exit(0)
 
 if ( 'MCMC' == solver ):
   print("[Korali] Running MCMC Plotter...")
   plot_mcmc(path, allFiles, live, generation, test)
   exit(0)

 if ( 'CMAES' == solver):
   print("[Korali] Running CMAES Plotter...")
   plot_cmaes(path, allFiles, live, generation, test, mean)
   exit(0)
  
 if ( 'CCMAES' == solver):
   print("[Korali] Running cCMAES Plotter...")
   plot_cmaes(path, allFiles, live, generation, test, mean)
   exit(0)

 if ( 'DEA' == solver ):
   print("[Korali] Running DEA Plotter...")
   plot_dea(path, allFiles, live, generation, test, mean)
   exit(0)

 print("[Korali] Error: Did not recognize method for plotting...")
 exit(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='korali.plotter', description='Process korali results in _korali_result (default) folder.')
    parser.add_argument('--dir', help='directory of result files', default='_korali_result', required = False)
    parser.add_argument('--all', help='plot all available results', action='store_true', required = False)
    parser.add_argument('--live', help='no auto close, keep polling for new result files', action='store_true', required = False)
    parser.add_argument('--generation', help='plot results of generation GENERATION', action='store', type=int, required = False)
    parser.add_argument('--mean', help='plot mean of objective variables', action='store_true', required = False)
    parser.add_argument('--check', help='verifies that korali.plotter is available', action='store_true', required = False)
    parser.add_argument('--test', help='run without graphics (for testing purpose)', action='store_true', required = False)
    args = parser.parse_args()

    main( args.dir, args.all, args.live, args.generation, args.mean, args.check, args.test)
