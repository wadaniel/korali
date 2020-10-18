//  Korali Environment for Nek5000
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "environment.hpp"
#include "unistd.h"
#include "stdio.h"

korali::Sample* sample;
double _reward;
double _state[STATE_SIZE];
double _action[ACTION_SIZE];

void runEnvironment(korali::Sample &s)
{
  // Storing sample pointer
  sample = &s;

  // Getting sample ID to create working environment
  size_t sampleId = s["Sample Id"];

  // Creating work environment
  char envdir[1024];
  sprintf(envdir, "_work%lu", sampleId);

  char command[1024];
  sprintf(command, "rm -rf %s", envdir); system(command);
  sprintf(command, "mkdir %s", envdir); system(command);
  sprintf(command, "cp _config/* %s", envdir); system(command);
  chdir(envdir);

  // Initializing environment
  resetenv_(); // Cleans Nek5000 global variables for re-run
  auto comm = MPI_COMM_WORLD;
  nek_init_(&comm); // When running with MPI, this should be the MPI team

  // Running environment
  nek_solve_();

  // Cleaning Environment
  nek_end_();
  chdir("..");
}

void updateAgent()
{
 // Setting states
 for (size_t i = 0; i < STATE_SIZE; i++)
  (*sample)["State"][i] = _state[i];

 // Setting Reward
 (*sample)["Reward"] = _reward;

 // Getting new action
 sample->update();

 // Reading new action
 for (size_t i = 0; i < ACTION_SIZE; i++)
  _action[i] = (*sample)["Action"][i];

 // Printing step info
 printf("[Korali] ----------------------------------------------------\n");
 printf("[Korali] Step Information:\n");
 printf("[Korali] State: [ %f, %f, %f, %f, %f, %f ]\n", _state[0], _state[1], _state[2], _state[3], _state[4], _state[5]);
 printf("[Korali] Reward: %f\n", _reward);
 printf("[Korali] Action: %f\n", _action[0]);
 printf("[Korali] ----------------------------------------------------\n");
}

