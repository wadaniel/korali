//  Korali environment for Nek5000
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "model.hpp"

void runEnvironment(korali::Sample &s)
{
  // Getting sample ID to create working environment
  size_t sampleId = s["Sample Id"];

  // Creating work environment
  createWorkEnvironment(sampleId);

  // Initializing State
  auto comm = MPI_COMM_WORLD;
  nek_init_(&comm); // When running with MPI, this should be the MPI team

  // Setting initial state (unknown for now)
  s["State"] = { 0.0, 0.0, 0.0, 0.0, 0.0, }; // getState();

  // Getting new action
  s.update();

  // Reading new action
  std::vector<double> action = s["Action"];

  // Printing Action:
  printf("Action: [ %f", action[0]);
  for (size_t i = 1; i < action.size(); i++) printf(", %f", action[i]);
  printf("]\n");

  // Setting action (not defined yet)
  //setAction()

  // Running simulation
  nek_solve_();

  // Storing reward (unknown for now)
  s["Reward"] = 0.0; // getReward();

  // Storing new state (unkown for now)
  s["State"] = { 0.0, 0.0, 0.0, 0.0, 0.0, };

  // Finalizing environment
  nek_end_();
}

void createWorkEnvironment(size_t envId)
{
 char envdir[1024];
 sprintf(envdir, "_env%6lu", envId);

 char command[1024];
 sprintf(command, "rm -r %s", envdir); system(command);
 sprintf(command, "mkdir %s", envdir); system(command);
 sprintf(command, "_model/turbChannel* %s", envdir); system(command);
}
