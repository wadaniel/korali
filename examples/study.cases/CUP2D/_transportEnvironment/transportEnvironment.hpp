//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>

void runEnvironment(korali::Sample &s);
void runEnvironmentCmaes(korali::Sample &s);
extern std::string _resultsPath;
extern int _argc;
extern char **_argv;

#include "Obstacles/SmartCylinder.h"
#include "Simulation.h"

void initializeEnvironment();
void setInitialConditions(SmartCylinder* agent, std::vector<double>& start, bool randomized);
bool isTerminal( SmartCylinder* agent, std::vector<double>& target );

// Global variables for the simulation (ideal if this would be a class instead)
extern std::mt19937 _randomGenerator;
extern Simulation *_environment;


// Helper functions
inline double distance(std::vector<double> x, std::vector<double> y) { return std::sqrt((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1])); }
