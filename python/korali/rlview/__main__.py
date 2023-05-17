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

import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")
sns.color_palette("tab10")

from korali.rlview.utils import get_figure

##################### Plotting Reward History
def plotRewardHistory( ax, results, averageDepth, showCI, showData, showObservations, featureReward, showAgents, maxExp, dir ):

    # get color
    color = next(ax._get_lines.prop_cycler)['color']

    maxEpisodes = math.inf

    returnsHistory = []
    observationHistory = []

    if showCI > 0.0:
        medianReturns  = []
        lowerCiReturns = []
        upperCiReturns = []
    else:
        meanReturns = []
        stdReturns = []

    numResults = len(results)
    if showAgents:
        if numResults != 1:
            print("Script only supports plotting the results of the individual agents for one run!")
            exit(-1)

        numResults = results[0]["Problem"]["Agents Per Environment"]
        
    ## Unpack and preprocess the results
    for r in results:
        # Load Returns
        if featureReward:
            returns = np.array(r["Solver"]["Training"]["Feature Reward History"])
        else:
            returns = np.array(r["Solver"]["Training"]["Reward History"])

        if (r["Problem"]["Agents Per Environment"] > 1) and not showAgents:
            returns = np.mean(returns, axis=1)
            returns = np.reshape(returns, (1,-1))

        returns = np.reshape(returns, (numResults,-1))

        for _return in returns:

            # Load and save cumulative sum of observations
            observationHistory.append(np.cumsum(r["Solver"]["Training"]["Experience History"]))

            # store results
            returnsHistory.append(_return)

            # Adjust x-range
            currEpisodeCount = len(_return)
            if (currEpisodeCount < maxEpisodes): maxEpisodes = currEpisodeCount

            if showCI > 0.0:
                median= [ _return[0] ]
                confIntervalLower= [ _return[0] ]
                confIntervalUpper= [ _return[0] ]

                for i in range(1, len(_return)):
                    # load data in averging window
                    startPos = max(i - averageDepth, 0)
                    endPos = i
                    data = _return[startPos:endPos]
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
                averageReturns = np.cumsum(_return)
                averageStart = averageReturns[:averageDepth]/np.arange(1,averageDepth+1)
                averageRest  = (averageReturns[averageDepth:]-averageReturns[:-averageDepth])/float(averageDepth)

                averageReturnsSquared = np.cumsum(_return*_return)
                averageSquaredStart = averageReturnsSquared[:averageDepth]/np.arange(1,averageDepth+1)
                averageSquaredRest  = (averageReturnsSquared[averageDepth:]-averageReturnsSquared[:-averageDepth])/float(averageDepth)

                # Append Results
                meanReturn = np.append(averageStart, averageRest)
                meanReturns.append( meanReturn )

                stdReturn = np.sqrt(np.append(averageSquaredStart, averageSquaredRest) - meanReturn**2)
                stdReturns.append(stdReturn)

    ## Only keep first maxEpisodes entries
    for i in range(numResults):
        observationHistory[i] = observationHistory[i][:maxEpisodes]
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
    if showObservations:
        episodes = observationHistory[0]
    
    if maxExp > 0:
        ex = np.linspace(0,5000000,num=len(episodes))
    else:
        ex = episodes

    if numResults == 1:
        if showCI > 0.0: # Plot median together with CI
            ax.plot(ex, medianReturns[0], '-', linewidth=2.0, zorder=1, label=dir, color=color)
            ax.fill_between(ex, lowerCiReturns[0], upperCiReturns[0], alpha=0.5)
        else: # .. or mean with standard deviation
            ax.plot(ex, meanReturns[0], '-', linewidth=2.0, zorder=1, label=dir, color=color)
            ax.fill_between(ex, meanReturns[0]-stdReturns[0], meanReturns[0]+stdReturns[0], alpha=0.2)
        if showData:
            ax.plot(ex, returnsHistory[i], 'x', markersize=1.3, linewidth=2.0, alpha=0.2, zorder=0, color=plt.gca().lines[-1].get_color())
    elif showAgents:
        for i in range(numResults):
            if showCI > 0.0:
                ax.plot(ex, medianReturns[i], '-', linewidth=2.0, zorder=1, label=dir, color=color)
                ax.fill_between(ex, lowerCiReturns[i], upperCiReturns[i], alpha=0.5)
            else:
                ax.plot(ex, meanReturns[i], '-', linewidth=2.0, zorder=1, label=dir, color=color)
                ax.fill_between(ex, meanReturns[i]-stdReturns[i], meanReturns[i]+stdReturns[i], alpha=0.5)
            if showData:
                ax.plot(ex, returnsHistory[i], 'x', markersize=1.3, linewidth=2.0, alpha=0.2, zorder=0, color=plt.gca().lines[-1].get_color())
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

            ax.plot(ex, median, '-', linewidth=2.0, zorder=1, label=dir)
            ax.fill_between(ex, confIntervalLower, confIntervalUpper, alpha=0.5)
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

            ax.plot(ex, mean, '-', linewidth=2.0, zorder=1, label=dir, color=color)
            ax.fill_between(ex, mean-std, mean+std, alpha=0.5)

        if showData:
            for i in range(len(returnsHistory)):
                ax.plot(ex, returnsHistory[i], 'x', markersize=1.3, linewidth=2.0, alpha=0.2, zorder=0, color=plt.gca().lines[-1].get_color())

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
        description='Plot the results of a Korali Reinforcement Learning execution. If single run, the displayed statistics are computed from the data in the averaging window. For multiple runs the displayed statistics are computed from the zeroth moments of the single runs.')
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
        type=float,
        default=+math.inf,
        required=False)
    parser.add_argument(
        '--maxReward',
        help='Maximum reward to display',
        type=float,
        default=-math.inf,
        required=False)
    parser.add_argument(
        '--featureReward',
        help='Plot Feature Reward',
        required=False,
        action='store_true')
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
        '--showCumulativeRewards',
        help='Option to show the cumulative reward for each episode.',
        action='store_true',
        required=False)
    parser.add_argument(
        '--showObservations',
        help='Option to show # Observations instead of # Episodes.',
        action='store_true',
        required=False)
    parser.add_argument(
        '--showLegend',
        help='Option to show the legend.',
        action='store_true',
        required=False)
    parser.add_argument(
        '--output',
        help='Indicates the output file path. If not specified, it prints to screen.',
        required=False)
    parser.add_argument(
        '--showAgents',
        help='Enable the plotting of the returns for each agent.',
        action='store_true',
        required=False)
    parser.add_argument(
      '--test',
      help='Run without graphics (for testing purpose)',
      action='store_true',
      required=False)
    parser.add_argument(
        '--maxExp',
        help='Number of exp on xaxis.',
        type=int,
        default=0,
        required=False)

    args = parser.parse_args()

    ### Validating arguments
    if args.showCI < 0.0 or args.showCI > 1.0:
        print("[Korali] Argument of confidence interval must be in [0,1].")
        exit(-1)

    ### Setup without graphics, if needed
    if (args.test or args.output): 
        matplotlib.use('Agg')
 
    ### Reading values from result files
    results = parseResults(args.dir, args.numRuns)

    ### Creating figure
    fig, ax = get_figure(width='article')

    ### Creating plot
    for run in range(len(results)):
        plotRewardHistory(ax, results[run], args.averageDepth, args.showCI, args.showCumulativeRewards, args.showObservations, args.featureReward, args.showAgents, args.maxExp, args.dir[run])

    """
    ax.set_ylabel('Cumulative Reward')
    if args.showObservations:
        ax.set_xlabel('# Observations')
    else:
        ax.set_xlabel('# Episodes')
    ax.set_title('Korali RL History Viewer')
    if args.showLegend:
        ax.legend()
    """

    if args.maxEpisodes < math.inf:
        ax.set_xlim([0, args.maxEpisodes])
    if (args.minReward < math.inf) and (args.maxReward > -math.inf):
        ax.set_ylim([args.minReward - 0.1*abs(args.minReward), args.maxReward + 0.1*abs(args.maxReward)])

    ### Show/save plot
    fig.tight_layout()
    if (args.output is None):
        plt.show()
    else:
        plt.savefig(args.output)
