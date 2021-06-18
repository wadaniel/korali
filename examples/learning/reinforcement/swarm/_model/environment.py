from swarm import *

def environment( args, s ):
    # set set parameters and initialize environment
    numIndividuals       = args["numIndividuals"]
    numTimesteps         = args["numTimesteps"]
    numNearestNeighbours = args["numNearestNeighbours"]
    sim = swarm( numIndividuals, numNearestNeighbours )

    # compute pair-wise distances and view-angles
    done = sim.preComputeStates()
    # set initial state
    state = []
    for i in np.arange(sim.N):
        state.append( sim.getState( i ) )
    s["State"] = state

    ## run simulation
    step = 0
    if done: 
        print("Initial configuration is terminal state...")
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

            # get reward
            if done:
                s["Reward"][i] = -10
            else:
                s["Reward"][i] = sim.getReward( i )

            # rotation in wished direction
            sim.fishes[i].updateDirection()
            # update positions
            sim.fishes[i].updateLocation()

        # compute pair-wise distances and view-angles
        done = sim.preComputeStates()
        # set state
        state = []
        for i in np.arange(sim.N):
            state.append( sim.getState(i) )
        s["State"] = state      
        
        step += 1

    # Setting termination status
    if done:
        s["Termination"] = "Terminal"
    else:
        s["Termination"] = "Truncated"