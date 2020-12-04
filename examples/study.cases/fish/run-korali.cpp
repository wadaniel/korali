#include "_environment/environment.hpp"
#include "korali.hpp"

int main(int argc, char *argv[])
{
// Gathering actual arguments from MPI
//MPI_Init(&argc, &argv);

// Initializing environment
#ifdef CUBISM
  initializeEnvironment(argc, argv);
#endif

  auto e = korali::Experiment();

  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Action Repeat"] = 1;
  e["Problem"]["Actions Between Policy Updates"] = 1;

  // Setting up the 16 state variables
  size_t curVariable = 0;
  for (; curVariable < 16; curVariable++)
  {
    e["Variables"][curVariable]["Name"] = std::string("StateVar") + std::to_string(curVariable);
    e["Variables"][curVariable]["Type"] = "State";
  }

  e["Variables"][curVariable]["Name"] = "Curvature";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -1.0;
  e["Variables"][curVariable]["Upper Bound"] = +1.0;
  e["Variables"][curVariable]["Exploration Sigma"] = 0.1;

  curVariable++;
  e["Variables"][curVariable]["Name"] = "Force";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -0.25;
  e["Variables"][curVariable]["Upper Bound"] = +0.25;
  e["Variables"][curVariable]["Exploration Sigma"] = 0.05;

  //// Defining Agent Configuration

  e["Solver"]["Type"] = "Agent / Continuous / GFPT";
  e["Solver"]["Optimization Steps Per Update"] = 1;
  e["Solver"]["Experiences Between Agent Trainings"] = 1;
  e["Solver"]["Cache Persistence"] = 10;

  e["Solver"]["Random Action Probability"]["Initial Value"] = 0.01;
  e["Solver"]["Random Action Probability"]["Target Value"] = 0.01;
  e["Solver"]["Random Action Probability"]["Decrease Rate"] = 0.00;

  e["Solver"]["Experience Replay"]["Start Size"] = 1000;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 100000;

  //// Defining Critic Configuration

  e["Solver"]["Critic"]["Learning Rate"] = 0.0001;
  e["Solver"]["Critic"]["Discount Factor"] = 0.99;
  e["Solver"]["Critic"]["Mini Batch Size"] = 128;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear";

  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 128;
  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh";

  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 128;
  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh";

  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear";

  //// Defining Policy Configuration

  e["Solver"]["Policy"]["Learning Rate"] = 0.000001;
  e["Solver"]["Policy"]["Mini Batch Size"] = 128;
  e["Solver"]["Policy"]["Target Accuracy"] = 0.0001;

  e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 128;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 128;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Tanh";

  ////// Defining Termination Criteria

  e["Solver"]["Training Reward Threshold"] = 100.0;
  e["Solver"]["Policy Testing Episodes"] = 20;
  e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 100.0;

  ////// If using syntax test, run for a couple generations only

#ifdef TEST
  e["Solver"]["Termination Criteria"]["Max Generations"] = 20;
#endif

  ////// Setting file output configuration

  e["File Output"]["Enabled"] = false;

  ////// Running Experiment

  auto k = korali::Engine();
  k.run(e);
}
