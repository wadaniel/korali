//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "swimmerEnvironment.hpp"
#include <chrono>
#include <filesystem>

int _argc;
char **_argv;

std::mt19937 _randomGenerator;
Simulation *_environment;

// Swimmer following an obstacle
void runEnvironment(korali::Sample &s)
{
  // Setting seed
  size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory
  char resDir[64];
  sprintf(resDir, "%s/sample%08lu", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), sampleId);
  if( not std::filesystem::create_directories(resDir) )
  {
    fprintf(stderr, "[Korali] Error creating results directory for environment\n");
    exit(-1);
  };

  // Redirecting all output to the log file
  char logFilePath[128];
  sprintf(logFilePath, "%s/log.txt", resDir);
  auto logFile = freopen(logFilePath, "a", stdout);
  if (logFile == NULL)
  {
    printf("[Korali] Error creating log file: %s.\n", logFilePath);
    exit(-1);
  }

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Obtaining environment objects and agent
  Shape *object = _environment->getShapes()[0];
  StefanFish *agent = dynamic_cast<StefanFish *>(_environment->getShapes()[1]);

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Reseting environment and setting initial conditions
  _environment->reset();
  setInitialConditions(agent, object, s["Mode"] == "Training"); 
  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();

  // Setting initial state
  auto state = agent->state(object);
  s["State"] = state;

  // Variables for time and step conditions
  double t = 0;        // Current time
  size_t curStep = 0;  // current Step
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action     

  // Setting maximum number of steps before truncation
  size_t maxSteps = 200;

  // Starting main environment loop
  bool done = false;
  while (done == false && curStep < maxSteps)
  {
    // Getting new action
    s.update();

    // Reading new action
    std::vector<double> action = s["Action"];

    // Setting action
    agent->act(t, action);

    // Run the simulation until next action is required
    dtAct = agent->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct && done == false )
    {
      // Compute timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;

      // Advance simulation
      _environment->advance(dt);

      // Check for terminal state.
      done = isTerminal(agent, object);
    }

    // Reward is -10 if state is terminal; otherwise obtain it from the agent's efficiency
    double reward = done ? -10.0 : agent->EffPDefBnd;

    // Obtaining new agent state
    state = agent->state(object);

    // Storing reward
    s["Reward"] = reward;

    // Storing new state
    s["State"] = state;

    // Printing Information:
    printf("[Korali] -------------------------------------------------------\n");
    printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    printf("[Korali] State: [ %.3f", state[0]);
    for (size_t i = 1; i < state.size(); i++) printf(", %.3f", state[i]);
    printf("]\n");
    printf("[Korali] Action: [ %.3f, %.3f ]\n", action[0], action[1]);
    printf("[Korali] Reward: %.3f\n", reward);
    printf("[Korali] Terminal?: %d\n", done);
    printf("[Korali] -------------------------------------------------------\n");
    fflush(stdout);

    // Write action to file
    ofstream myfile ("actions.txt", ios::out | ios::app );
    if (myfile.is_open())
    {
      myfile << action[0] << " " << action[1] << std::endl;
      myfile.close();
    }
    else{
      fprintf(stderr, "Unable to open actions.txt file\n");
      exit(-1);
    }

    // Advancing to next step
    curStep++;
  }

  // Flush CUP logger
  logger.flush();

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

void setInitialConditions(StefanFish *agent, Shape *object, const bool isTraining)
{
  // Initial fixed conditions
  double initialAngle = 0.0;
  std::vector<double> initialPosition{0.9,1.0};

  // with noise
  if (isTraining)
  {
    std::uniform_real_distribution<double> disA(-10. / 180. * M_PI, 10. / 180. * M_PI);
    std::uniform_real_distribution<double> disX(-0.1, 0.1);
    std::uniform_real_distribution<double> disY(-0.05, 0.05);

    initialAngle = disA(_randomGenerator);
    initialPosition[0] = initialPosition[0] + disX(_randomGenerator);
    initialPosition[1] = initialPosition[1] + disY(_randomGenerator);
  }

  printf("[Korali] Initial Condition:\n");
  printf("[Korali] angle: %f\n", initialAngle);
  printf("[Korali] x: %f\n", initialPosition[0]);
  printf("[Korali] y: %f\n", initialPosition[1]);

  // Write initial condition to file
  ofstream myfile ("initialCondition.txt");
  if (myfile.is_open())
  {
    myfile << initialAngle << " " << initialPosition[0] << " " << initialPosition[1] << std::endl;
    myfile.close();
  }
  else{
    fprintf(stderr, "Unable to open initialCondition.txt file\n");
    exit(-1);
  }

  // Setting initial position and orientation for the fish
  agent->setCenterOfMass(initialPosition.data());
  agent->setOrientation(initialAngle);
}

bool isTerminal(StefanFish *agent, Shape *object)
{
  const double X = (agent->center[0] - object->center[0]);
  const double Y = (agent->center[1] - object->center[1]);

  bool terminal = false;
  if (X < +0.15) terminal = true;
  if (X > +0.55) terminal = true;
  if (Y < -0.1) terminal = true;
  if (Y > +0.1) terminal = true;

  return terminal;
}
