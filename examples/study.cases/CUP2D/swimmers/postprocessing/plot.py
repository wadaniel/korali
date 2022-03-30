#! /usr/bin/env python3

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
import korali

import colorsys
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], min(1, amount * c[1]), c[2])

import glob

def loadTestingReturns( prefix, path ):
    # Go to target directory and get list with wished files
    os.chdir(path)
    files = sorted(glob.glob("{}*".format(prefix)))

    # Iterate over found files and extract testing returns
    returns = []
    for file in files:
        current = []
        with open(file, 'r') as read_obj:
            for line in read_obj:
                if "[Korali]    + (Average) Cumulative Reward:" in line:
                    current.append(float(line[-10:]))
        returns.append(current)

    print("returns:", returns)
    # return matrix
    return returns

def plotTestReturn():
    radiusHalfdisk = 0.03 + np.arange(0,10)*(0.07-0.03)/9.0
    returnsHalfDisk = loadTestingReturns( "halfDisk_largeDomain.eval_out_", "/project/s929/pweber/karmanPaper/runs/halfDisk_largeDomain.eval")
    # returnsHalfDisk = [ [71.877472, 35.746422, 72.399307, 70.539886, 67.355606, 71.559250, 66.029266, 20.292776, 72.102036, 74.534355], 
    #                     [47.281033, 29.244068, 42.671959, 36.717030, 76.072678, 72.767601, 52.312687, 68.508308, 14.979681, 70.792526], 
    #                     [70.061028, 14.720476, 27.964424, 77.676506, 31.307430, 40.412399, 70.584953, 72.067741, 29.980824, 14.374123], 
    #                     [69.693024,  7.931906, 36.665131, 47.728031, 70.362495, 23.927502, 76.204643, 77.182266, 43.064526, 31.136116], 
    #                     [58.347542, 34.805363, 76.137146, 47.171993, 72.005035, 53.830025, 72.698814, 40.248714, 22.150364, 49.365700], 
    #                     [76.023628, 24.614338, 25.675465, 74.114304, 22.584686, 76.594612, 76.488304, 27.150826, 71.926483, 26.284504], 
    #                     [40.906456, 32.818642, 12.491627, 13.860413, 40.230045, 77.034988, 31.331985, 14.530348, 31.037430, 30.154568], 
    #                     [71.273689, 33.176735, 46.527603, 11.925661, 20.736364, 21.980368,  7.083330, 13.439631,  8.955492, 36.891823], 
    #                     [71.538788,  7.513134, 25.847656, 15.441429, 51.058018, 21.440580, 71.967415,  6.614027, 10.408041, 18.553194], 
    #                     [28.699856, 26.573059, 14.584204, 38.553646, 25.892330, 19.607376, 37.387802, 13.230320,  9.246807, 73.480888]
    #                   ]

    frequencyHydrofoil = 0.28 + np.arange(0,10)*(1.37-0.28)/9.0
    returnsHydrofoil = loadTestingReturns( "hydrofoil_largeDomain.eval_out_", "/project/s929/pweber/karmanPaper/runs/hydrofoil_largeDomain.eval")
    # returnsHydrofoil = [ [ 7.524055,  8.761213,  7.830694,  9.644695,  9.341175,  9.586660,  8.080988,  9.722359,  8.511307,  9.050045], 
    #                      [16.472881, 17.260635, 17.213276, 17.836643, 17.454916, 17.335394, 18.742882, 17.821896, 16.891821, 18.241610], 
    #                      [27.796574, 28.365913, 30.191246, 31.387844, 31.079899, 30.861362, 29.572357, 30.686298, 30.204124, 30.853065], 
    #                      [31.885803, 30.937855, 34.244350, 33.140045, 37.171410, 33.289799, 36.103275, 34.446991, 29.897125, 31.002312], 
    #                      [34.539356, 36.893112, 35.360149, 38.029636, 34.759682, 39.640270, 40.136360, 32.067318, 38.879211, 36.932201], 
    #                      [41.775311, 44.016697, 43.247524, 43.832706, 43.082180, 44.469143, 43.745361, 42.071392, 45.290058, 43.442963], 
    #                      [46.519913, 56.936756, 60.087769, 58.383892, 58.635048, 56.661800, 58.144440, 59.124634, 43.803516, 41.188057], 
    #                      [59.414246, 59.849018, 22.273815, 58.199020, 26.052868, 59.797054, 59.453354, 33.913383, 60.048744, 59.533871], 
    #                      [39.231834, 63.176430, 27.476566, 63.709484, -3.826483, 48.245983, 48.591808, 27.453629, 63.832481, 27.544292], 
    #                      [60.062302, 59.399956, 60.432770, 59.051277, 59.129372, 60.556324, 59.666649, 59.538666, 62.303299, 60.172390]
    #                     ]

    ## CAREFUL, HYDROFOIL FIRST!!
    # returnsMultitask = loadTestingReturns( "multitask", "/project/s929/pweber/karmanPaper/runs/multitask_largeDomain.eval")
    # returnsMultitask = [ [ 7.840683,  9.300875,  8.329866,  8.723188,  8.442017,  8.982254,  8.347025,  9.448149,  9.649590,  7.742252], 
    #                      [13.404787, 13.842098, 12.187981, 14.141544, 14.565126, 12.941868, 14.339552, 13.613255, 13.093306, 12.441664], 
    #                      [22.794262, 22.200760, 22.682598, 24.128239, 22.595558, 22.919006, 22.207226, 19.990974, 22.377518, 22.045048], 
    #                      [26.538879, 26.902885, 29.596600, 24.886024, 27.646160, 29.824348, 28.094971, 28.786110, 32.526493, 30.041107], 
    #                      [29.359322, 22.244526, 26.519573, 30.150570, 29.827305, 26.044811, 28.683334, 28.913582, 28.936703, 33.751942], 
    #                      [30.698437, 36.423939, 34.631737, 37.971977, 36.101154, 34.100414, 37.018822, 28.709339, 32.935959, 34.453701], 
    #                      [42.687531, 35.514561, 33.741890, 34.020992, 32.782013, 33.931095, 26.653896, 38.582424, 38.942169, 35.485458], 
    #                      [21.096457, 24.599312, 37.933212, 35.788063, 20.835583, 33.700233, 38.993019, 17.687687, 32.360115, 41.854683], 
    #                      [62.633438, 33.595375, 34.152256, 17.020424, 20.862793, 44.161411, 62.181736, 62.558620, 40.624359, 59.515076], 
    #                      [57.755352, 59.597820, 60.669189, 59.048481, 59.731812, 59.183178, 59.054337, 38.987343, 60.587223, 58.705013]
                        #   , 
                        #  ###########
                        #   [], 
                        #   [], 
                        #   [], 
                        #   [], 
                        #   [], 
                        #   [], 
                        #   [], 
                        #   [], 
                        #   [], 
                        #   []
                        # ]

    fig, axs = plt.subplots(1, 2, figsize=(6,3), dpi=100, sharey=True)
    axs[0].plot(radiusHalfdisk, np.mean(returnsHalfDisk, axis=1), color="C0")
    # axs[0].plot(radiusHalfdisk, returnsMultitask[:10], color="C2")
    axs[0].set_xlabel('radius')
    axs[0].set_ylabel('testing return')
    axs[0].grid(True, which='minor')

    axs[1].plot(frequencyHydrofoil, np.mean(returnsHydrofoil, axis=1), color="C1")
    # axs[1].plot(frequencyHydrofoil, np.mean(returnsMultitask, axis=1)[:10], color="C2")
    axs[1].set_xlabel('frequency')
    axs[1].grid(True, which='minor')

    plt.show()

def parseRM( dir ):

    # Load from Folder containing Results
    configFile = dir + '/latest'
    if (not os.path.isfile(configFile)):
        print("[Korali] Error: Did not find any results in the {0}...".format(configFile))
        exit(-1)

    with open(configFile) as f:
        config = json.load(f)

    dataFile = dir + '/state.json'
    if (not os.path.isfile(dataFile)):
        print("[Korali] Error: Did not find any results in the {0}...".format(dataFile))
        exit(-1)

    with open(dataFile) as f:
        data = json.load(f)

    return data, config

def parseActions( dir, numTests ):

    # Load from Folder containing Results
    data = []
    for s in range(numTests):
        actionFile = str(dir) + "/../_testingResults/sample{:03d}/actions0.txt".format(s)
        if (not os.path.isfile(actionFile)):
            print("[Korali] Error: Did not find any results in {0}...".format(actionFile))
            exit(-1)

        with open(actionFile) as f:
            data.append(np.loadtxt(actionFile))

    return data

def environment( data, config, xlim, ylim, N, s ):
    aspect = ylim / xlim
    xCoords = np.linspace(0,xlim,N)
    yCoords = np.linspace(0,ylim,int(aspect*N)+1)

    averageValue = np.zeros((N,int(aspect*N)+1))
    averagedPolicyParameter = np.zeros((N,int(aspect*N)+1))

    # Containers to store state values and policy parameters
    states = []
    stateValues = []
    policyParams  = []
    replayMemory = data["Experience Replay"]
    for experience in replayMemory:
        state = np.array(experience["State"]).reshape((-1,)).tolist()
        states.append(state)
        s["State"]  = state
        s["Reward"] = 0

        # Getting new action
        s.update()

        # Append state value and distribution parameters
        stateValues.append(s["Policy"]["State Value"][0])
        policyParams.append(s["Policy"]["Distribution Parameters"][0])

    s["State"]  = state
    s["Reward"] = 0
    s["Termination"] = "Terminal"

    # need numpy array for the following
    states = np.array(states)
    stateValues = np.array(stateValues)
    policyParams  = np.array(policyParams)

    print("states:", states)
    print("stateValues", stateValues)
    print("policyParams", policyParams)

    # Get state rescaling factor and undo scaling
    rescalingMeans = np.array(config["Solver"]["State Rescaling"]["Means"])
    rescalingSigmas = np.array(config["Solver"]["State Rescaling"]["Sigmas"])

    rescalingMeans = np.reshape(rescalingMeans,(-1,))
    rescalingSigmas = np.reshape(rescalingSigmas,(-1,))

    stateX = 0.9 + ( states[:,0] * rescalingSigmas[0] + rescalingMeans[0] ) * 0.2
    stateY = 0.5 + ( states[:,1] * rescalingSigmas[1] + rescalingMeans[1] ) * 0.2

    for i in range(len(xCoords)-1):
        for j in range(len(yCoords)-1):
            indices = ( stateX >= xCoords[i] ) & ( stateX < xCoords[i+1] ) \
                    & ( stateY >= yCoords[j] ) & ( stateY < yCoords[j+1] )
            averageValue[i,j] = np.mean(stateValues[indices])
    np.savetxt("stateValues.txt", averageValue)

    filenames = [ [ "meanAction1", "meanAction2"], [ "stdAction1", "stdAction2"] ]
    for l in range(2): #mean/sigma
        for k in range(2): #components
            for i in range(len(xCoords)-1):
                for j in range(len(yCoords)-1):
                        indices = ( stateX >= xCoords[i] ) & ( stateX < xCoords[i+1] ) \
                                & ( stateY >= yCoords[j] ) & ( stateY < yCoords[j+1] )
                        averagedPolicyParameter[i,j] = np.mean(policyParams[indices,l*2+k])
            np.savetxt(filenames[l][k]+".txt", averagedPolicyParameter)

def evaluateNeuralNetwork( data, config, xlim, ylim, N, path ):
    # Create Korali Engine and Experiment
    k = korali.Engine()
    e = korali.Experiment()

    # Load Experiment from file
    found = e.loadState("_trainingResults/latest")
    if found == True: 
        print("[Korali] Evaluation results found...\n")
    else:
        print(stderr, "[Korali] Error: cannot find previous results\n")
        exit(0)

    # Configure Korali
    e["Problem"]["Environment Function"] = lambda x : environment( data, config, xlim, ylim, N, x );
    e["File Output"]["Path"] = "_trainingResults";
    e["Solver"]["Mode"] = "Testing";

    # random seeds for environment
    e["Solver"]["Testing"]["Sample Ids"] = [ 42 ];

    # launch testing
    k.run(e);

def plotValueFunctionRM( ax, data, config, xlim, ylim, N ):
    # Binning of x- and y- coordines
    aspect = ylim / xlim
    xCoords = np.linspace(0,xlim,N)
    yCoords = np.linspace(0,ylim,int(aspect*N)+1)
    averageValue = np.empty((N,int(aspect*N)+1))
    averageValue.fill(np.nan)

    # Fill state and state-value vector with data from replay memory
    states = []
    stateValues = []
    replayMemory = data["Experience Replay"]
    for experience in replayMemory:
        states.append(experience["State"])
        stateValues.append(experience["State Value"])
    states = np.array(states)
    stateValues = np.array(stateValues)

    # Get state rescaling factor and undo scaling
    rescalingMeans = np.array(config["Solver"]["State Rescaling"]["Means"])
    rescalingSigmas = np.array(config["Solver"]["State Rescaling"]["Sigmas"])

    rescalingMeans = np.reshape(rescalingMeans,(-1,))
    rescalingSigmas = np.reshape(rescalingSigmas,(-1,))

    states = np.reshape(states, (-1,states.shape[2]))
    stateX = 0.9 + ( states[:,0] * rescalingSigmas[0] + rescalingMeans[0] ) * 0.2
    stateY = 0.5 + ( states[:,1] * rescalingSigmas[1] + rescalingMeans[1] ) * 0.2

    for i in range(len(xCoords)-1):
        for j in range(len(yCoords)-1):
            indices = ( stateX >= xCoords[i] ) & ( stateX < xCoords[i+1] ) \
                    & ( stateY >= yCoords[j] ) & ( stateY < yCoords[j+1] )
            averageValue[i,j] = np.mean(stateValues[indices])

    X, Y = np.meshgrid(xCoords, yCoords, indexing='ij')

    im = ax.pcolormesh(X, Y, averageValue, cmap='viridis')
    ax.set_xlim([0, xlim])
    ax.set_ylim([0, ylim])
    fig.colorbar(im, ax=ax)

    # Add circle
    circle = plt.Circle((0.6, 0.5), 0.05, color='gray')
    ax.add_patch(circle)


def plotPolicyRM( ax, data, config, xlim, ylim, N ):
    # Binning of x- and y- coordines
    aspect = ylim / xlim
    xCoords = np.linspace(0,xlim,N)
    yCoords = np.linspace(0,ylim,int(aspect*N)+1)
    averagedPolicyParameter = np.zeros((N,int(aspect*N)+1))
    averagedPolicyParameter.fill(np.nan)

    # Fill state and state-value vector with data from replay memory
    states = []
    policyParams  = []
    replayMemory = data["Experience Replay"]
    for experience in replayMemory:
        states.append(experience["State"])
        policyParams.append(experience["Current Policy"]["Distribution Parameters"])
    states = np.array(states)
    policyParams  = np.array(policyParams)

    policyParams = np.reshape(policyParams,(-1,policyParams.shape[2]))

    # Get state rescaling factor and undo scaling
    rescalingMeans = np.array(config["Solver"]["State Rescaling"]["Means"])
    rescalingSigmas = np.array(config["Solver"]["State Rescaling"]["Sigmas"])

    rescalingMeans = np.reshape(rescalingMeans,(-1,))
    rescalingSigmas = np.reshape(rescalingSigmas,(-1,))

    states = np.reshape(states, (-1,states.shape[2]))
    stateX = 0.9 + ( states[:,0] * rescalingSigmas[0] + rescalingMeans[0] ) * 0.2
    stateY = 0.5 + ( states[:,1] * rescalingSigmas[1] + rescalingMeans[1] ) * 0.2

    for l in range(2): #mean/sigma
        for k in range(2): #components
            for i in range(len(xCoords)-1):
                for j in range(len(yCoords)-1):
                        indices = ( stateX >= xCoords[i] ) & ( stateX < xCoords[i+1] ) \
                                & ( stateY >= yCoords[j] ) & ( stateY < yCoords[j+1] )
                        averagedPolicyParameter[i,j] = np.mean(policyParams[indices,l*2+k])

            X, Y = np.meshgrid(xCoords, yCoords, indexing='ij')

            axs[l][k].set_xlim([0, xlim])
            axs[l][k].set_ylim([0, ylim])
            axs[l][k].set_xlabel('x')
            axs[l][k].set_ylabel('y')
            axs[l][k].grid(True, which='minor')
            if l == 0 and k == 0:
                im = ax[l][k].pcolormesh(X, Y, averagedPolicyParameter, cmap='RdBu', vmin=-1, vmax=1)
            elif l == 0 and k == 1:
                im = ax[l][k].pcolormesh(X, Y, averagedPolicyParameter, cmap='RdBu')
            else:
                im = ax[l][k].pcolormesh(X, Y, averagedPolicyParameter, cmap='viridis')
            
            fig.colorbar(im, ax=ax[l][k])

            # Add circle
            circle = plt.Circle((0.6, 0.5), 0.05, color='gray')
            axs[l][k].add_patch(circle)

def plotValue( averageValue, xlim, ylim, N ):
    aspect = ylim / xlim
    xCoords = np.linspace(0,xlim,N)
    yCoords = np.linspace(0,ylim,int(aspect*N)+1)

    X, Y = np.meshgrid(xCoords, yCoords, indexing='ij')

    im = ax.pcolormesh(X, Y, averageValue, cmap='viridis')
    fig.colorbar(im, ax=ax)

    # Add circle
    circle = plt.Circle((0.6, 0.5), 0.05, color='gray')
    ax.add_patch(circle) 

def plotPolicy( l, k, averagedPolicyParameter, xlim, ylim, N ):
    aspect = ylim / xlim
    xCoords = np.linspace(0,xlim,N)
    yCoords = np.linspace(0,ylim,int(aspect*N)+1)

    X, Y = np.meshgrid(xCoords, yCoords, indexing='ij')

    if l == 0 and k == 0:
        im = ax[l][k].pcolormesh(X, Y, averagedPolicyParameter, cmap='RdBu', vmin=-1, vmax=1)
    elif l == 0 and k == 1:
        im = ax[l][k].pcolormesh(X, Y, averagedPolicyParameter, cmap='RdBu')
    else:
        im = ax[l][k].pcolormesh(X, Y, averagedPolicyParameter, cmap='viridis')
    
    fig.colorbar(im, ax=ax[l][k])

    # Add circle
    circle = plt.Circle((0.6, 0.5), 0.05, color='gray')
    axs[l][k].add_patch(circle)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='rl.statistics',
        description='Plot the statistics of the replay memory.')
    parser.add_argument(
        '--dir',
        help='Path to result files, separated by space',
        type = str,
        default = "",
        required=False)
    parser.add_argument(
        '--output',
        help='Indicates the output file path. If not specified, it prints to screen.',
        required=False)
    parser.add_argument(
        '--xlim',
        help='maximal extent in x-direction.',
        type = float,
        default = 2,
        required=False)
    parser.add_argument(
        '--ylim',
        help='maximal extent in y-direction.',
        type = float,
        default = 1,
        required=False)
    parser.add_argument(
        '--N',
        help='Number of points in x-direction.',
        type = int,
        default = 100,
        required=False)
    parser.add_argument(
        '--type',
        help='type of plot to create.',
        type = str,
        default = "",
        required=False)
    parser.add_argument(
        '--plotTestingReturns',
        help='Option to show testing returns.',
        action='store_true',
        required=False)
    parser.add_argument(
        '--numTests',
        help='Number of test runs performed.',
        type = int,
        default=0,
        required=False)

    args = parser.parse_args()

    if args.plotTestingReturns:
        plotTestReturn()

    if args.numTests > 0:
        data = parseActions( args.dir, args.numTests )

        fig, ax = plt.subplots(1, 1, figsize=(12,6))

        for i, d in enumerate(data):
            ax.plot(d[:,0], d[:,1], color=lighten_color("C0",0.1+0.1*i), linewidth=2)
            ax.set_ylim([-1, 1])
            ax.set_xlabel('time t')
            ax.set_ylabel('action')


    if args.type != "":
        data, config = parseRM(args.dir)

    if args.type == "value":
        ### Creating figure
        fig, ax = plt.subplots(1, 1, figsize=(6,3))

        ax.set_xlim([0, args.xlim])
        ax.set_ylim([0, args.ylim])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, which='minor')
        plotValueFunctionRM( ax, data, config, args.xlim, args.ylim, args.N )
    elif args.type == "policy":
        ### Creating figure
        fig, axs = plt.subplots(2, 2, figsize=(12,6))

        plotPolicyRM( axs, data, config, args.xlim, args.ylim, args.N )
    elif args.type == "forward":
        evaluateNeuralNetwork( data, config, args.xlim, args.ylim, args.N, args.dir )

        ### Creating figure
        fig, ax = plt.subplots(1, 1, figsize=(12,6))

        ax.set_xlim([0, args.xlim])
        ax.set_ylim([0, args.ylim])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, which='minor')

        averageValue = np.loadtxt("stateValues.txt", args.xlim, args.ylim, args.N )
        plotValue( averageValue )

        ### Creating figure
        fig, axs = plt.subplots(2, 2, figsize=(12,6))

        filenames = [ [ "meanAction1", "meanAction2"], [ "stdAction1", "stdAction2"] ]
        for l in range(2): #mean/sigma
            for k in range(2): #components
                averagedPolicyParameter = np.loadtxt(filenames[l][k]+".txt")
                plotPolicy( l, k, averagedPolicyParameter, args.xlim, args.ylim, args.N )

    ### Show/save plot
    fig.tight_layout()
    if (args.output is None):
        plt.show()
    else:
        plt.savefig(args.output)