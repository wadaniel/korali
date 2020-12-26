#include "_environment/environment.hpp"
#include "korali.hpp"

std::string _resultsPath;

int main(int argc, char *argv[])
{
  // Gathering actual arguments from MPI

#ifndef TEST
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided != MPI_THREAD_FUNNELED)
  {
    printf("Error initializing MPI\n");
    exit(-1);
  }
#endif

  // Storing parameters
  _argc = argc;
  _argv = argv;

  // Getting number of workers
  int N = 1;
#ifndef TEST
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine

  // Initializing Cubism
  _environment = new Simulation(_argc, _argv);
  _environment->init();
#endif

  // Setting results path
  _resultsPath = "_results";

  // Creating Experiment
  auto e = korali::Experiment();

  // Configuring Experiment

  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Training Reward Threshold"] = 100.0;
  e["Problem"]["Policy Testing Episodes"] = 5;
  e["Problem"]["Actions Between Policy Updates"] = 1;

  ////// Checking if existing results are there and continuing them

  auto found = e.loadState(_resultsPath + std::string("/latest"));
  if (found == true) printf("Continuing execution from previous run...\n");

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
  e["Solver"]["Agent Count"] = N;
  e["Solver"]["Experiences Per Generation"] = N * 4;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Cache Persistence"] = 10;

  e["Solver"]["Experience Replay"]["Start Size"] = 1000;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 100000;
  e["Solver"]["Experience Replay"]["Serialization Frequency"] = 1;

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

  e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 1000.0;

  ////// If using syntax test, run for a couple generations only

#ifdef TEST
  e["Solver"]["Termination Criteria"]["Max Generations"] = 20;
#endif

  ////// Setting results output configuration

  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Path"] = _resultsPath;

  ////// Running Experiment

  auto k = korali::Engine();

  // Configuring profiler output

  k["Profiling"]["Detail"] = "Full";
  k["Profiling"]["Path"] = _resultsPath + std::string("/profiling.json");
  k["Profiling"]["Frequency"] = 60;

#ifndef TEST
  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Communicator"] = MPI_COMM_WORLD;
#endif

  k.run(e);
}
