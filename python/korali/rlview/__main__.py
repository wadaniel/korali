#! /usr/bin/env python3
import os
import sys
import signal
import json
import argparse
import time
import matplotlib
import importlib
import math 
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from korali.plot.helpers import hlsColors, drawMulticoloredLine
from scipy.signal import savgol_filter
import pdb

##################### Plotting Reward History

def plotRewardHistory( ax, colorIndx, results, averageDepth, showCI, showData ):
    
    maxEpisodes = math.inf

    returnsHistory = []

    if showCI > 0.0:
        medianReturns  = []
        lowerCiReturns = []
        upperCiReturns = []
    else:
        meanReturns = []
        stdReturns = []

    ## Unpack and preprocess the results
    for r in results:
        # Load Returns
        returns = np.array(r["Solver"]["Training"]["Reward History"])
        if r["Problem"]["Agents Per Environment"] != 1:
            returns = np.mean(returns,axis=0)

        # store results
        returnsHistory.append(returns)

        # Adjust x-range
        currEpisodeCount = len(returns)
        if (currEpisodeCount < maxEpisodes): maxEpisodes = currEpisodeCount

        if showCI > 0.0:
            median= [ returns[0] ]
            confIntervalLower= [ returns[0] ]
            confIntervalUpper= [ returns[0] ]

            for i in range(1, len(returns)):
                # load data in averging window
                startPos = max(i - averageDepth, 0)
                endPos = i
                data = returns[startPos:endPos]
                # compute quantiles
                median.append(np.percentile(data, 50))
                confIntervalLower.append( np.percentile(data, 50-50*showCI) )
                confIntervalUpper.append( np.percentile(data, 50+50*showCI) )
            
            # append data
            medianReturns.append(median)
            lowerCiReturns.append(confIntervalLower)
            upperCiReturns.append(confIntervalUpper)
        else:
            # Average returns over averageDepth episodes
            averageReturns = np.cumsum(returns)
            averageStart = averageReturns[:averageDepth]/np.arange(1,averageDepth+1)
            averageRest  = (averageReturns[averageDepth:]-averageReturns[:-averageDepth])/float(averageDepth)

            averageReturnsSquared = np.cumsum(returns*returns)
            averageSquaredStart = averageReturnsSquared[:averageDepth]/np.arange(1,averageDepth+1)
            averageSquaredRest  = (averageReturnsSquared[averageDepth:]-averageReturnsSquared[:-averageDepth])/float(averageDepth)

            # Append Results
            meanReturn = np.append(averageStart, averageRest)
            meanReturns.append( meanReturn )

            stdReturn = np.append(averageSquaredStart, averageSquaredRest) - meanReturn**2
            stdReturns.append(stdReturn)

    ## Only keep first maxEpisodes entries
    for i, res in enumerate(results):
        returnsHistory[i] = returnsHistory[i][:maxEpisodes]
        if showCI > 0.0:
            medianReturns[i]  = medianReturns[i][:maxEpisodes]
            lowerCiReturns[i] = lowerCiReturns[i][:maxEpisodes]
            upperCiReturns[i] = upperCiReturns[i][:maxEpisodes]
        else:
            meanReturns[i] = meanReturns[i][:maxEpisodes]
            stdReturns[i] = stdReturns[i][:maxEpisodes]

    ## Plot results
    episodes = np.arange(1,maxEpisodes+1)
    if showData:
        for i in range(len(returnsHistory)):
            ax.plot(episodes, returnsHistory[i], 'x', markersize=1.3, color=cmap(colorIndx), lineWidth=1.5, alpha=0.15, zorder=0)
    if len(results) == 1:
        if showCI > 0.0: # Plot median together with CI
            ax.plot(episodes, medianReturns[0], '-', color=cmap(colorIndx), lineWidth=3.0, zorder=1) 
            ax.fill_between(episodes, lowerCiReturns[0], upperCiReturns[0][:maxEpisodes], color=cmap(colorIndx), alpha=0.2)
        else: # .. or mean with standard deviation
            ax.plot(episodes, meanReturns[0], '-', color=cmap(colorIndx), lineWidth=3.0, zorder=1) 
            ax.fill_between(episodes, meanReturns[0]-stdReturns[0], meanReturns[0]+stdReturns[0], color=cmap(colorIndx), alpha=0.2)
    else:
        if showCI > 0.0: # Plot median over runs
            medianReturns = np.array(medianReturns)

            median = []
            confIntervalLower = []
            confIntervalUpper = []
            for i in range(maxEpisodes):
                # load data
                data = medianReturns[:,i]
                # compute quantiles
                median.append( np.percentile(data, 50) )
                confIntervalLower.append( np.percentile(data, 50-50*showCI) )
                confIntervalUpper.append( np.percentile(data, 50+50*showCI) )

            ax.plot(episodes, median, '-', color=cmap(colorIndx), lineWidth=3.0, zorder=1) 
            ax.fill_between(episodes, confIntervalLower, confIntervalUpper, color=cmap(colorIndx), alpha=0.2)
        else: # .. or mean with standard deviation
            meanReturns = np.array(meanReturns)

            mean = []
            std  = []
            for i in range(maxEpisodes):
                # load data
                data = meanReturns[:,i]
                # compute mean and standard deviation
                mean.append( np.mean(data) )
                std.append( np.std(data) )
            mean = np.array(mean)
            std  = np.array(std)

            ax.plot(episodes, mean, '-', color=cmap(colorIndx), lineWidth=3.0, zorder=1) 
            ax.fill_between(episodes, mean-std, mean+std, color=cmap(colorIndx), alpha=0.2)
      
    ax.set_ylabel('Cumulative Reward')
    ax.set_xlabel('# Observations')
    ax.set_title('Korali RL History Viewer')
 

##################### Results parser

def parseResults( dir, numRuns ):
    # Empty Container for results
    results = [ ]

    # Load from Folder containing Results
    for p in dir:
        result = [ ]
        # Load result for each run
        for run in range(numRuns):
            configFile = p + '/latest'
            if numRuns > 1:
                configFile = p + "/run{}".format(run) + '/latest'
            if (not os.path.isfile(configFile)):
                print("[Korali] Error: Did not find any results in the {0} folder...".format(configFile))
                exit(-1)
            with open(configFile) as f:
                data = json.load(f)
            result.append(data)
        results.append(result)
  
    return results
 
##################### Main Routine: Parsing arguments and result files
  
if __name__ == '__main__':

    # Setting termination signal handler
    signal.signal(signal.SIGINT, lambda x, y: exit(0))

    # Parsing arguments
    parser = argparse.ArgumentParser(
        prog='korali.rlview',
        description='Plot the results of a Korali Reinforcement Learning execution.')
    parser.add_argument(
        '--dir',
        help='Path(s) to result files, separated by space',
        required=True,
        nargs='+')
    parser.add_argument(
        '--maxEpisodes',
        help='Maximum number of episodes (x-axis) to display',
        type=int,
        default=+math.inf,
        required=False)
    parser.add_argument(
        '--minReward',
        help='Minimum reward to display',
        default=+math.inf,
        required=False)
    parser.add_argument(
        '--maxReward',
        help='Maximum reward to display',
        default=-math.inf,
        required=False)
    parser.add_argument(
        '--averageDepth',
        help='Specifies the number of episodes used to compute statistics',
        type=int,
        default=100,
        required=False)
    parser.add_argument(
        '--numRuns',
        help='Number of evaluation runs that are stored under --dir/runXX.',
        type=int,
        default=1,
        required=False)
    parser.add_argument(
        '--showCI',
        help='Option to plot median+CI (default=False -> plot mean+std).',
        type = float,
        default=0.0,
        required=False)
    parser.add_argument(
        '--showData',
        help='Option to plot datapoints.',
        action='store_true',
        required=False)
    parser.add_argument(
        '--output',
        help='Indicates the output file path. If not specified, it prints to screen.',
        required=False)

    args = parser.parse_args()

    ### Validating arguments
    if args.showCI < 0.0 or args.showCI > 1.0:
        print("[Korali] Argument of confidence interval must be in [0,1].")
        exit(-1)

    if args.output:
        if not (output.endswith(".png") or output.endswith(".eps") or output.endswith(".svg")):
            print("[Korali] Error: Outputfile '{0}' must end with '.eps', '.png' or '.svg' suffix.".format(output))
            sys.exit(-1)
        matplotlib.use('Agg')
 
    ### Reading values from result files
    results = parseResults(args.dir, args.numRuns)
 
    ### Creating figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
     
    ### Creating plot
    colorIndx = 0.0
    cmap = matplotlib.cm.get_cmap('brg')

    for run in range(len(results)):
        colorIndx = run / float(len(results)-1+1e-10)
        plotRewardHistory(ax, colorIndx, results[run], args.averageDepth, args.showCI, args.showData)

    ax.set_ylabel('Cumulative Reward')  
    ax.set_xlabel('# Observations')
    ax.set_title('Korali RL History Viewer')

    ax.yaxis.grid()
    if args.maxEpisodes < math.inf:
        ax.set_xlim([0, args.maxEpisodes-1])
    if (args.minReward < math.inf) and (args.maxReward > -math.inf):
        ax.set_ylim([args.minReward - 0.1*abs(args.minReward), args.maxReward + 0.1*abs(args.maxReward)])

    ### Show/save plot
    if (args.output is None):
        plt.show()
    else:
        plt.savefig(args.output)

