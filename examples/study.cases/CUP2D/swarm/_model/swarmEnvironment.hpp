//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>
#include "Obstacles/StefanFish.h"
#include "Simulation.h"
#include "Utils/BufferedLogger.h"

// Settings for different setups
#define NAGENTS 4
#define XMAX 1.4
#define YMIN 0.7
#define YMAX 1.3

// #define NAGENTS 9
// #define XMAX 2
// #define YMIN 0.6
// #define YMAX 1.4

// #define NAGENTS 16
// #define XMAX 2.6
// #define YMIN 0.5
// #define YMAX 1.5

// #define NAGENTS 25
// #define XMAX 3.2
// #define YMIN 0.4
// #define YMAX 1.6

// command line arguments are read in Korali application
extern int _argc;
extern char **_argv;

void runEnvironment(korali::Sample &s);
void setInitialConditions(StefanFish *agent, size_t agentId, const bool isTraining);
bool isTerminal(StefanFish *agent);
