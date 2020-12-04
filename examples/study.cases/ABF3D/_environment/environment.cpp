//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "environment.hpp"

#ifdef MSODE

bool _isTraining;
std::mt19937 _randomGenerator;
std::unique_ptr<msode::rl::MSodeEnvironment> _environment;

void runEnvironment(korali::Sample &s)
{
  // Setting seed
  size_t sampleId = s["Sample Id"];

  // Reseting environment and setting initial conditions
  _environment->reset(_randomGenerator, sampleId, true);

  // Setting initial state
  s["State"] = _environment->getState();

  // Defining status variable that tells us whether when the simulation is done
  Status status{Status::Running};

  // Storing action index
  size_t curActionIndex = 0;

  // Starting main environment loop
  while (status == Status::Running)
  {
    // Getting new action
    s.update();

    // Reading new action
    std::vector<double> action = s["Action"];

    // Printing Action:
    // if (curActionIndex % 20 == 0)
    // {
    //  printf("Action %lu: [ %f", curActionIndex, action[0]);
    //  for (size_t i = 1; i < action.size(); i++) printf(", %f", action[i]);
    //  printf("]\n");
    // }

    // Setting action
    status = _environment->advance(action);

    // Storing reward
    s["Reward"] = _environment->getReward();

    // Storing new state
    s["State"] = _environment->getState();

    // Increasing action count
    curActionIndex++;
  }

  // Setting finalization status
  if (status == Status::Success)
    s["Termination"] = "Terminal";
  else
    s["Termination"] = "Truncated";
}

void initializeEnvironment(const std::string confFileName)
{
  std::ifstream confFile(confFileName);

  if (!confFile.is_open())
  {
    fprintf(stderr, "Could not open the config file '%s'", confFileName.c_str());
    exit(-1);
  }

  const Config config = json::parse(confFile);
  _environment = rl::factory::createEnvironment(config, ConfPointer(""));
}

#else

// Environment for syntax test only
void runEnvironment(korali::Sample &s)
{
  fprintf(stderr, "[Warning] Using test-only setup. If you want to run the actual experiment, run ./install_deps.sh first and re-compile.\n");

  for (size_t i = 0; i < 243; i++)
  {
    s["State"] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    s.update();
    s["Reward"] = 0.0001;
  }
  s["Termination"] = "Terminal";
}

#endif
