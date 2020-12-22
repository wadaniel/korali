#include "_environment/environment.hpp"
#include "korali.hpp"

int main(int argc, char *argv[])
{
#ifndef TEST
  initializeEnvironment("_deps/msode/launch_scripts/rl/config/helix_2d_eu_const.json");
#endif

  auto e = korali::Experiment();

  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Training Reward Threshold"] = 1.0;
  e["Problem"]["Policy Testing Episodes"] = 20;
  e["Problem"]["Actions Between Policy Updates"] = 1;

  ////// Checking if existing results are there and continuing them

  auto found = e.loadState("_results/latest");
  if (found == true) printf("Continuing execution from previous run...\n");

  //// Setting state variables

  e["Variables"][0]["Name"] = "Swimmer 1 - Pos X";
  e["Variables"][1]["Name"] = "Swimmer 1 - Pos Y";
  e["Variables"][2]["Name"] = "Swimmer 1 - Pos Z";
  e["Variables"][3]["Name"] = "Swimmer 1 - Quaternion X";
  e["Variables"][4]["Name"] = "Swimmer 1 - Quaternion Y";
  e["Variables"][5]["Name"] = "Swimmer 1 - Quaternion Z";
  e["Variables"][6]["Name"] = "Swimmer 1 - Quaternion W";
  e["Variables"][7]["Name"] = "Swimmer 2 - Pos X";
  e["Variables"][8]["Name"] = "Swimmer 2 - Pos Y";
  e["Variables"][9]["Name"] = "Swimmer 2 - Pos Z";
  e["Variables"][10]["Name"] = "Swimmer 2 - Quaternion X";
  e["Variables"][11]["Name"] = "Swimmer 2 - Quaternion Y";
  e["Variables"][12]["Name"] = "Swimmer 2 - Quaternion Z";
  e["Variables"][13]["Name"] = "Swimmer 2 - Quaternion W";

  //// Setting action variables

#ifndef TEST
  auto [lowerBounds, upperBounds] = _environment->getActionBounds();
#else
  std::vector<float> lowerBounds = {0.0, 0.0, 0.0, 0.0};
  std::vector<float> upperBounds = {1.0, 1.0, 1.0, 1.0};
#endif

  e["Variables"][14]["Name"] = "Frequency (w)";
  e["Variables"][14]["Type"] = "Action";
  e["Variables"][14]["Lower Bound"] = lowerBounds[0];
  e["Variables"][14]["Upper Bound"] = upperBounds[0];
  e["Variables"][14]["Exploration Sigma"] = (upperBounds[0] - lowerBounds[0]) * 0.1;

  e["Variables"][15]["Name"] = "Rotation X";
  e["Variables"][15]["Type"] = "Action";
  e["Variables"][15]["Lower Bound"] = lowerBounds[1];
  e["Variables"][15]["Upper Bound"] = upperBounds[1];
  e["Variables"][15]["Exploration Sigma"] = (upperBounds[1] - lowerBounds[1]) * 0.1;

  e["Variables"][16]["Name"] = "Rotation Y";
  e["Variables"][16]["Type"] = "Action";
  e["Variables"][16]["Lower Bound"] = lowerBounds[2];
  e["Variables"][16]["Upper Bound"] = upperBounds[2];
  e["Variables"][16]["Exploration Sigma"] = (upperBounds[2] - lowerBounds[2]) * 0.1;

  e["Variables"][17]["Name"] = "Rotation Z";
  e["Variables"][17]["Type"] = "Action";
  e["Variables"][17]["Lower Bound"] = lowerBounds[3];
  e["Variables"][17]["Upper Bound"] = upperBounds[3];
  e["Variables"][17]["Exploration Sigma"] = (upperBounds[3] - lowerBounds[3]) * 0.1;

  //// Defining Agent Configuration

  e["Solver"]["Type"] = "Agent / Continuous / GFPT";
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Agent Count"] = 1;
  e["Solver"]["Experiences Per Generation"] = 972;
  e["Solver"]["Experiences Between Policy Updates"] = 10;
  e["Solver"]["Cache Persistence"] = 10;
  e["Solver"]["Discount Factor"] = 0.99;
  e["Solver"]["Mini Batch Size"] = 256;

  e["Solver"]["Experience Replay"]["Start Size"] = 1000;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 262144;
  e["Solver"]["Experience Replay"]["Serialization Frequency"] = 20;
  e["Solver"]["Mini Batch Strategy"] = "Uniform";

  //// Defining Critic/Policy Configuration

  e["Solver"]["Critic"]["Learning Rate"] = 0.0001;
  e["Solver"]["Policy"]["Learning Rate"] = 0.000001;
  e["Solver"]["Policy"]["Target Accuracy"] = 0.0001;
  e["Solver"]["Policy"]["Optimization Candidates"] = 8;

  //// Defining Neural Network

  e["Solver"]["Neural Network"]["Engine"] = "OneDNN";

  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128;

  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128;

  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";

  ////// Defining Termination Criteria

  e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 1.3;

  ////// If using configuration test, run for a couple generations only

#ifdef TEST
  e["Solver"]["Termination Criteria"]["Max Generations"] = 5;
#endif

  ////// Setting file output configuration

  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 20;
  e["File Output"]["Path"] = "_results";

  auto k = korali::Engine();
  k.run(e);
}
