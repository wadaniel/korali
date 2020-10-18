//  Korali model for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <random>

void runEnvironment(korali::Sample &s);

extern "C"
{
 void nek_init_(int* comm);
 void nek_solve_();
 void nek_end_();
 void updateagent_(double* state, double* reward, double* action);
}

void updateAgent(double* state, double reward, double* action);
void updateagent_(double* state, double* reward, double* action) { updateAgent(state, *reward, action); }

extern korali::Sample* sample;
