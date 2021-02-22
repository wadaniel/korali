//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "transportEnvironment.hpp"
#include <chrono>
#include <filesystem>

int _argc;
char **_argv;

Simulation *_environment;
std::mt19937 _randomGenerator;

// Swimmer following an obstacle
void runEnvironment(korali::Sample &s)
{
  // Setting seed
  size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory
  char resDir[64];
  sprintf(resDir, "%s/sample%08lu", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), sampleId);
  std::filesystem::create_directories(resDir);

  // Redirecting all output to the log file
  char logFilePath[128];
  sprintf(logFilePath, "%s/log.txt", resDir);
  auto logFile = freopen(logFilePath, "a", stdout);
  if (logFile == NULL)
  {
    printf("Error creating log file: %s.\n", logFilePath);
    exit(-1);
  }

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Obtaining agent
  SmartCylinder* agent = dynamic_cast<SmartCylinder *>(_environment->getShapes()[0]);

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Reseting environment and setting initial conditions
  _environment->resetRL();
  setInitialConditions(agent, s["Mode"] == "Training");

  // Set target 
  std::vector<double> target{0.8,0.5};

  // Setting initial state
  auto state = agent->state( target );
  s["State"] = state;

  // Setting initial time and step conditions
  double t = 0;        // Current time
  double tNextAct = 0; // Time until next action
  size_t curStep = 0;  // current Step

  // Setting maximum number of steps before truncation
  size_t maxSteps = 200;

  // Starting main environment loop
  bool done = false;
  while (done == false && curStep < maxSteps)
  {
    // Getting initial time
    auto beginTime = std::chrono::steady_clock::now(); // Profiling

    // Getting new action
    s.update();

    // Reading new action
    std::vector<double> action = s["Action"];

    // Setting action
    agent->act( action );

    // Run the simulation until next action is required
    tNextAct += 0.1;
    while ( t < tNextAct )
    {
      // Advance simulation
      const double dt = _environment->calcMaxTimestep();
      t += dt;

      // Advance simulation and check whether it is correct
      if (_environment->advance(dt))
      {
        fprintf(stderr, "Error during environment\n");
        exit(-1);
      }

      // Re-check if simulation is done.
      done = isTerminal( agent, target );
    }

    // Reward is +10 if state is terminal; otherwise obtain it from inverse distance to target
    double reward = done ? 100.0 : agent->reward( target );

    // Getting ending time
    auto endTime = std::chrono::steady_clock::now(); // Profiling
    double actionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count() / 1.0e+9;

    // Printing Information:
    printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    printf("[Korali] State: [ %.3f", state[0]);
    for (size_t i = 1; i < state.size(); i++) printf(", %.3f", state[i]);
    printf("]\n");
    printf("[Korali] Action: [ %.3f, %.3f ]\n", action[0], action[1]);
    printf("[Korali] Reward: %.3f\n", reward);
    printf("[Korali] Terminal?: %d\n", done);
    printf("[Korali] Time: %.3fs\n", actionTime);
    printf("[Korali] -------------------------------------------------------\n");
    fflush(stdout);

    // Obtaining new agent state
    state = agent->state( target );

    // Storing reward
    s["Reward"] = reward;

    // Storing new state
    s["State"] = state;

    // Advancing to next step
    curStep++;
  }

  // Setting finalization status
  if (done == true)
    s["Termination"] = "Terminal";
  else
    s["Termination"] = "Truncated";

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);
}

void setInitialConditions(SmartCylinder* agent, const bool isTraining)
{
  // Initial fixed conditions
  double locationX = 0.2;
  double locationY = 0.5;

  // or with noise
  if (isTraining)
  {
    std::uniform_real_distribution<double> dis(-0.01, 0.01);

    double distX = dis(_randomGenerator);
    double distY = dis(_randomGenerator);

    locationX += distX;
    locationY += distY;
  }

  printf("[Korali] Initial Conditions:\n");
  printf("[Korali] locationX: %f\n", locationX);
  printf("[Korali] locationY: %f\n", locationY);

  // Setting initial position and orientation for the fish
  double C[2] = { locationX, locationY};
  agent->setCenterOfMass(C);

  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();
}

bool isTerminal(SmartCylinder* agent, std::vector<double> target )
{
  const double dX = (agent->center[0] - target[0]);
  const double dY = (agent->center[1] - target[1]);

  const double dTarget = std::sqrt(dX*dX+dY*dY);

  bool terminal = false;
  if ( dTarget < 1e-1 ) terminal = true;

  return terminal;
}
