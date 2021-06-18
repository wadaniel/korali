from swarm import *

def getState( i, distancesMat, anglesMat, directionMat):
    # get array for agent i
    distances = distancesMat[i,:]
    angles    = anglesMat[i,:]
    directions= directionMat[i,:,:]
    # sort and select nearest neighbours
    idSorted = np.argsort( distances )
    idNearestNeighbours = idSorted[:numNearestNeighbours]
    distancesNearestNeighbours = distances[ idNearestNeighbours ]
    anglesNearestNeighbours = angles[ idNearestNeighbours ]
    directionNearestNeighbours = directions[idNearestNeighbours,:]
    # the state is the distance (or direction?) and angle to the nearest neigbours
    return np.array([ distancesNearestNeighbours, anglesNearestNeighbours ]).flatten().tolist() # or np.array([ directionNearestNeighbours, anglesNearestNeighbours ]).flatten()

def environment( args, s ):

    numIndividuals       = args["numIndividuals"]
    numTimesteps         = args["numTimesteps"]
    numNearestNeighbours = args["numNearestNeighbours"]
    sim = swarm( numIndividuals )

    # compute pair-wise distances and view-angles
    distancesMat, anglesMat, directionMat = sim.computeStates()
    # set initial state
    state = []
    for i in np.arange(sim.N):
        state.append( getState(i, distancesMat, anglesMat, directionMat) )
    s["State"] = state

    ## run simulation
    step = 0
    done = False
    while (step < numTimesteps) and (not done):
        # Getting new action
        s.update()

        ## apply action and advance environment
        for i in np.arange(sim.N):
            polarAngles = s["Action"][i]
            x = np.cos(polarAngles[0])*np.sin(polarAngles[1])
            y = np.sin(polarAngles[0])*np.sin(polarAngles[1])
            z = np.cos(polarAngles[1])
            sim.fishes[i].wishedDirection = [ x, y, z ]

            # Termination state in case distance matrix has entries < eps
            if (distances < sim.fishes[i].sigmaPotential ).any():
                done = True

            # get reward
            if done:
                s["Reward"][i] = -10
            else:
                s["Reward"][i] = sim.fishes[i].getReward(distancesNearestNeighbours)

            # rotation in wished direction
            sim.fishes[i].updateDirection()
            # update positions
            sim.fishes[i].updateLocation()

        # compute pair-wise distances and view-angles
        distancesMat, anglesMat, directionMat = sim.computeStates()
        # set state
        state = []
        for i in np.arange(sim.N):
            state.append( getState(i, distancesMat, anglesMat, directionMat) )
        s["State"] = state      
        
        step += 1

    # Setting termination status
    if done:
        s["Termination"] = "Terminal"
    else:
        s["Termination"] = "Truncated"