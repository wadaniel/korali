//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>
#include "Obstacles/StefanFish.h"
#include "Simulation.h"
#include "Utils/BufferedLogger.h"

// command line arguments are read in Korali application
extern int _argc;
extern char **_argv;

void runEnvironment(korali::Sample &s);
void setInitialConditions(StefanFish *agent, size_t agentId, const bool isTraining);
bool isTerminal(StefanFish *agent, size_t nAgents);
