//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "environment.hpp"

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

    // Scaling to lower-upper bounds
    auto [lowerBounds, upperBounds] = _environment->getActionBounds();
    for (size_t i = 0; i < action.size(); i++)
     action[i] = action[i] * (upperBounds[i] - lowerBounds[i]);

//     // Printing Action:
//     if (curActionIndex % 20 == 0)
//     {
//      printf("Action %lu: [ %f", curActionIndex, action[0]);
//      for (size_t i = 1; i < action.size(); i++) printf(", %f", action[i]);
//      printf("]\n");
//     }

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
