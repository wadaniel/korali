import argparse
import math
from pathlib import Path
from source.swarm import *
from source.plotter import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--visualize', help='whether to plot the swarm or not', required=True, type=int)
    parser.add_argument('--numIndividuals', help='number of fish', required=True, type=int)
    parser.add_argument('--numTimesteps', help='number of timesteps to simulate', required=True, type=int)
    parser.add_argument('--numNearestNeighbours', help='number of nearest neighbours used for state/reward', required=True, type=int)

    args = vars(parser.parse_args())

    numIndividuals       = args["numIndividuals"]
    numTimeSteps         = args["numTimesteps"]
    numNearestNeighbours = args["numNearestNeighbours"]
    assert numIndividuals > numNearestNeighbours, print("numIndividuals must be bigger than numNearestNeighbours")
    
    sim  = swarm( numIndividuals )
    step = 0
    action = [ 1, 0, 0 ]
    while step < numTimeSteps:
        print("timestep {}/{}".format(step+1, numTimeSteps))
        # if enable, plot current configuration
        if args["visualize"]:
            Path("./figures").mkdir(parents=True, exist_ok=True)
            # fixed camera
            plotSwarm( sim, step )
            # camera following center of swarm
            # plotSwarmCentered( sim, step )

        # compute pair-wise distances and view-angles
        distancesMat, anglesMat = sim.computeStates()

        # update swimming directions
        for i in np.arange(sim.N):
            print("agent {}/{}".format(i+1, sim.N))
            # get row giving distances / angle to other swimmers
            # TODO?: Termination state in case distance matrix has entries < eps
            distances = distancesMat[i,:]
            angles    = anglesMat[i,:]
            # sort and select nearest neighbours
            idSorted = np.argsort( distances )
            idNearestNeighbours = idSorted[:numNearestNeighbours]
            distancesNearestNeighbours = distances[ idNearestNeighbours ]
            anglesNearestNeighbours = angles[ idNearestNeighbours ]
            # the state is the distance and angle to the nearest neigbours
            state  = np.array([ distancesNearestNeighbours, anglesNearestNeighbours ]).flatten()
            print("state:", state)
            # set action
            assert len(action) == 3, print("action assumed to be 3D")
            assert math.isclose( np.linalg.norm(action),  1.0 ), print("action assumed to be normal vector")
            sim.fishes[i].wishedDirection = action
            # get reward
            reward = sim.fishes[i].getReward(distancesNearestNeighbours)
            print("reward:", reward)
            # rotation in wished direction
            sim.fishes[i].updateDirection()
            # update positions
            sim.fishes[i].updateLocation()

        step += 1