//  Korali environment for Nek5000
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "model.hpp"
#include "unistd.h"
#include "stdio.h"

korali::Sample* sample;

void runEnvironment(korali::Sample &s)
{
  // Storing sample pointer
  sample = &s;

  // Switching to work directory
  chdir("_work");

  // Initializing environment
  auto comm = MPI_COMM_WORLD;
  nek_init_(&comm); // When running with MPI, this should be the MPI team

  // Running environment
  nek_solve_();

  // Cleaning Environment
  nek_end_();
  chdir("..");
}

void updateAgent(double* state, double reward, double* action)
{
 // Setting states
 (*sample)["State"][0] = state[0];
 (*sample)["State"][1] = state[1];
 (*sample)["State"][2] = state[2];
 (*sample)["State"][3] = state[3];
 (*sample)["State"][4] = state[4];
 (*sample)["State"][5] = state[5];

 // Setting Reward
 (*sample)["Reward"] = reward;

 // Getting new action
 sample->update();

 // Reading new action
 action[0] = (*sample)["Action"][0];

 // Printing step info
 printf("State: [ %f, %f, %f, %f, %f, %f ]", state[0], state[1], state[2], state[3], state[4], state[5]);
 printf("Reward: %f", reward);
 printf("Action: %f", action[0]);
}

