//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "swarmEnvironment.hpp"
#include <chrono>
#include <filesystem>

int _argc;
char **_argv;
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

  // Creating simulation environment
  Simulation *_environment = new Simulation(_argc, _argv);
  _environment->init();

  // Obtaining agents
  std::vector<StefanFish *> agents(NAGENTS);
  for( size_t i = 0; i<NAGENTS; i++ )
    agents[i] = dynamic_cast<StefanFish *>(_environment->getShapes()[i]);

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Resetting environment and setting initial conditions
  for( size_t i = 0; i<NAGENTS; i++ )
    setInitialConditions(agents[i], i, s["Mode"] == "Training");
  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();

  // Setting initial state
  std::vector<std::vector<double>> states(NAGENTS);
  for( size_t i = 0; i<NAGENTS; i++ )
    states[i] = agents[i]->state();
  s["State"] = states;

  // Variables for time and step conditions
  double t = 0;        // Current time
  size_t curStep = 0;  // current Step
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action      

  // Setting maximum number of steps before truncation
  size_t maxSteps = 200;

  // Starting main environment loop
  bool done = false;
  while ( curStep < maxSteps && done == false )
  {
    // Getting new actions
    s.update();

    // Reading new actions
    auto actions = s["Action"];

    // Setting action for each agent
    for( size_t i = 0; i<NAGENTS; i++ )
    {
      std::vector<double> action = actions[i];

      // Write action to file
      std::stringstream filename;
      filename<<"actions_"<<i<<".txt";
      ofstream myfile(filename.str().c_str());
      if (myfile.is_open())
      {
        myfile << action[0] << " " << action[1] << std::endl;
        myfile.close();
      }
      else{
        fprintf(stderr, "Unable to open %s file...\n", filename.str().c_str());
        exit(-1);
      }

      // Apply action
      agents[i]->act(t, action);
    }

    // Run the simulation until next action is required
    dtAct = 0.;
    for( size_t i = 0; i<NAGENTS; i++ )
      if( dtAct < agents[i]->getLearnTPeriod() * 0.5 )
        dtAct = agents[i]->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct && done == false )
    {
      // Compute timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;

      // Advance simulation
      _environment->advance(dt);

      // Check if there was a collision -> termination.
      done = _environment->sim.bCollision;

      // Check termination because leaving margins
      for( size_t i = 0; i<NAGENTS; i++ )
        done = ( done || isTerminal(agents[i]) );
    }

    // Get state and action
    std::vector<double> rewards(NAGENTS);
    for( size_t i = 0; i<NAGENTS; i++ )
    {
      states[i]  = agents[i]->state();
      rewards[i] = done ? -10.0 : agents[i]->EffPDefBnd;
    }

    // Store state and action
    s["State"]  = states;
    s["Reward"] = rewards;
    
    // Printing Information:
    printf("[Korali] -------------------------------------------------------\n");
    printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    for( size_t i = 0; i<NAGENTS; i++ )
    {
      auto state = states[i];
      printf("[Korali] AGENT %ld\n", i);
      printf("[Korali] State: [ %.3f", state[0]);
      for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
      printf("]\n");
      printf("[Korali] Action: [ %.3f, %.3f ]\n", actions[i][0].get<float>(), actions[i][1].get<float>());
      printf("[Korali] Reward: %.3f\n", rewards[i]);
      printf("[Korali] Terminal?: %d\n", done);
      printf("[Korali] -------------------------------------------------------\n");
    }
    fflush(stdout);

    // Advancing to next step
    curStep++;
  }

  // Flush CUP logger
  logger.flush();

  // delete simulation class
  delete _environment;

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

void setInitialConditions(StefanFish *agent, size_t agentId, const bool isTraining)
{
  // Initial fixed conditions
  double initialAngle = 0.0;
  std::vector<double> initialPosition{ agent->origC[0], agent->origC[1] };

  // with noise
  if (isTraining)
  {
    std::uniform_real_distribution<double> disA(-5. / 180. * M_PI, 5. / 180. * M_PI);
    std::uniform_real_distribution<double> disX(-0.025, 0.025);
    std::uniform_real_distribution<double> disY(-0.05, 0.05);

    initialAngle = disA(_randomGenerator);
    initialPosition[0] = initialPosition[0] + disX(_randomGenerator);
    initialPosition[1] = initialPosition[1] + disY(_randomGenerator);
  }

  printf("[Korali] Initial Condition Agent %ld:\n", agentId);
  printf("[Korali] angle: %f\n", initialAngle);
  printf("[Korali] x: %f\n", initialPosition[0]);
  printf("[Korali] y: %f\n", initialPosition[1]);

  // Write initial condition to file
  std::stringstream filename;
  filename<<"initialCondition_"<<agentId<<".txt";
  ofstream myfile(filename.str().c_str());
  if (myfile.is_open())
  {
    myfile << initialAngle << " " << initialPosition[0] << " " << initialPosition[1] << std::endl;
    myfile.close();
  }
  else{
    fprintf(stderr, "Unable to open %s file...\n", filename.str().c_str());
    exit(-1);
  }

  // Setting initial position and orientation for the fish
  agent->setCenterOfMass(initialPosition.data());
  agent->setOrientation(initialAngle);
}

bool isTerminal(StefanFish *agent)
{
  const double X = agent->center[0];
  const double Y = agent->center[1];

  bool terminal = false;
  if (X < 0.4) terminal = true;
  if (X > XMAX) terminal = true;
  if (Y < YMIN) terminal = true;
  if (Y > YMAX) terminal = true;

  return terminal;
}
