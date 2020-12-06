//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "environment.hpp"
#include <filesystem>
#include <chrono>

#ifndef TEST

std::mt19937 _randomGenerator;
int _argc;
char** _argv;

void runEnvironment(korali::Sample &s)
{
  // Setting seed
  size_t sampleId  = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory
  char resDir[64];
  sprintf(resDir, "_results/sample%06lu", sampleId);
  if (std::filesystem::create_directories(resDir) == false)
   { printf("Error creating results directory: %s.\n", resDir); exit(-1); }

  // Redirecting all output to the log file
  char logFilePath[128];
  sprintf(logFilePath, "%s/log.txt", resDir);
  auto logFile = freopen(logFilePath, "a", stdout);
  if (logFile == NULL) 
   { printf("Error creating log file: %s.\n", logFilePath); exit(-1); }

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Initializing environment
  Simulation environment(_argc, _argv);
  environment.init();

  // Obtaining environment objects and agent
  Shape *object = environment.getShapes()[0];
  StefanFish *agent = dynamic_cast<StefanFish *>(environment.getShapes()[1]);
  
  // Reseting environment and setting initial conditions
  environment.reset();
  setInitialConditions(agent, object);

  // Setting initial state
  auto state = agent->state(object);
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
    agent->act(t, action);

    // Check if simulation is done.
    done = isTerminal(agent, object);

    // Run the simulation until next action is required
    tNextAct += agent->getLearnTPeriod() * 0.5;
    while (done == false && t < tNextAct)
    {
      // Advance simulation
      const double dt = environment.calcMaxTimestep();
      t += dt;

      // Advance simulation and check whether it is correct
      if (environment.advance(dt))
      {
        fprintf(stderr, "Error during environment\n");
        exit(-1);
      }

      // Re-check if simulation is done.
      done = isTerminal(agent, object);
    }

    // Reward is -10 if state is terminal; otherwise obtain it from the agent's efficiency
    double reward = done ? -10.0 : agent->EffPDefBnd;

    // Getting ending time
    auto endTime = std::chrono::steady_clock::now(); // Profiling
    double actionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count() / 1.0e+9; 

    // Printing Information:
    printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    printf("[Korali] State: [ %.3f", state[0]); 
    for (size_t i = 1; i < state.size(); i++) printf(", %.3f", state[i]);
    printf("]\n");
    printf("[Korali] Action: [ %.3f, %.3f ]\n", action[0], action[1]);
    printf("[Korali] Reward: %.3f\n",reward);
    printf("[Korali] Terminal?: %d\n", done);
    printf("[Korali] Time: %.3fs\n", actionTime);
    printf("[Korali] -------------------------------------------------------\n");
    fflush(stdout);

    // Obtaining new agent state
    state = agent->state(object);

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

void setInitialConditions(StefanFish *a, Shape *p)
{
  std::uniform_real_distribution<double> disA(-20. / 180. * M_PI, 20. / 180. * M_PI);
  std::uniform_real_distribution<double> disX(0.0, 0.5), disY(-0.25, 0.25);

  const double SX = disX(_randomGenerator);
  const double SY = disY(_randomGenerator);
  const double SA = disA(_randomGenerator);

  double C[2] = {p->center[0] + (1 + SX) * a->length, p->center[1] + SY * a->length};
  p->centerOfMass[1] = p->center[1] - (C[1] - p->center[1]);
  p->center[1] = p->center[1] - (C[1] - p->center[1]);
  a->setCenterOfMass(C);
  a->setOrientation(SA);
}

bool isTerminal(StefanFish *a, Shape *p)
{
  const double X = (a->center[0] - p->center[0]) / a->length;
  const double Y = (a->center[1] - p->center[1]) / a->length;
  assert(X > 0);
  // cylFollow
  return std::fabs(Y) > 1 || X < 0.5 || X > 3;
  // extended follow
  // return std::fabs(Y)>1 || X<1 || X>3;
  // restricted follow
  //return std::fabs(Y) > 0.75 || X < 1 || X > 2;
}

#else

// Environment for configuration test only
void runEnvironment(korali::Sample &s)
{
  fprintf(stderr, "[Warning] Using test-only setup. If you want to run the actual experiment, run ./install_deps.sh first and re-compile.\n");

  for (size_t i = 0; i < 10; i++)
  {
    s["State"] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    s.update();
    s["Reward"] = -10.0;
  }
  s["Termination"] = "Normal";
}

#endif
