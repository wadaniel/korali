//  Korali model for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include "Simulation.h"
#include "Obstacles/StefanFish.h"

void fishFollow(korali::Sample &s)
{
 int myRank, rankCount;
 MPI_Comm comm = korali::getKoraliMPIComm();
 MPI_Comm_rank(comm, &myRank);
 MPI_Comm_size(comm, &rankCount);

 size_t seed = s["Sample Id"];
 s["State"] = { 0.0 }; // Get initial state
 bool done = false; // flag to check whether simulation is done

 while( done == false )
 {
  // Getting new action
  s.update();

  // Reading new action
  std::vector<double> action = s["Action"][0];

  // Performing action
  //state, reward, done, info = cart.step(action)
  double reward = 0.0;
  std::vector<double> state = { 1.0 };

  // Storing reward
  s["Reward"] = reward;

  // Storing new state
  s["State"] = { state };

  // Check termination
  done = true;
 }
}
