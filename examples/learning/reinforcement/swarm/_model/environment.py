from swarm import *

def environment( args, s ):

    numIndividuals       = args["numIndividuals"]
    numTimeSteps         = args["numTimesteps"]
    numNearestNeighbours = args["numNearestNeighbours"]

    sim = swarm( numIndividuals )

    ## get initial state
    
    # compute pair-wise distances and view-angles
    distancesMat, anglesMat, directionMat = sim.computeStates()
    for i in np.arange(sim.N):
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
        # TODO: s["State"] = np.array([ distancesNearestNeighbours, anglesNearestNeighbours ]).flatten() or s["State"] = np.array([ directionNearestNeighbours, anglesNearestNeighbours ]).flatten()

    ## run simulation
    step = 0
    while step < numTimeSteps:
        # Getting new action
        # TODO: s.update()

        ## apply action and advance environment
        for i in np.arange(sim.N):
            # TODO: polarAngles = s["Action"]
            x = np.cos(polarAngles[0])*np.sin(polarAngles[1])
            y = np.sin(polarAngles[0])*np.sin(polarAngles[1])
            z = np.cos(polarAngles[1])
            sim.fishes[i].wishedDirection = [ x, y, z ]

            # get reward
            # TODO: s["Reward"] = sim.fishes[i].getReward(distancesNearestNeighbours)

            # rotation in wished direction
            sim.fishes[i].updateDirection()
            # update positions
            sim.fishes[i].updateLocation()

        ## get new state
        # compute pair-wise distances and view-angles
        distancesMat, anglesMat, directionMat = sim.computeStates()

        for i in np.arange(sim.N):
            # get row giving distances / angle to other swimmers
            # TODO?: Termination state in case distance matrix has entries < eps
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
            # TODO: s["State"] = np.array([ distancesNearestNeighbours, anglesNearestNeighbours ]).flatten() or s["State"] = np.array([ directionNearestNeighbours, anglesNearestNeighbours ]).flatten()
            
        step += 1

    # TODO?: Termination in case distance matrix has entries < eps
    s["Termination"] = "Truncated"