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
  std::string trainingResultsPath = "_trainingResults";
  std::string testingResultsPath = "_testingResults";

  // Creating Experiment
  auto e = korali::Experiment();
  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";

  ////// Checking if existing results are there and continuing them

#ifndef TEST
  auto found = e.loadState(trainingResultsPath + std::string("/latest"));
  if (found == true) printf("Continuing execution from previous run...\n");
#endif

  // Configuring Experiment
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Training Reward Threshold"] = 2.0;
  e["Problem"]["Policy Testing Episodes"] = 5;
  e["Problem"]["Actions Between Policy Updates"] = 100;

  // Adding custom setting to run the environment without dumping the state files during training
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.0;
  e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;

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
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Agent Count"] = N;
  e["Solver"]["Experiences Per Generation"] = N * 4;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Cache Persistence"] = 10;

  e["Solver"]["Experience Replay"]["Start Size"] = 1000;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 100000;
  e["Solver"]["Experience Replay"]["Serialization Frequency"] = 1;

  //// Defining Critic/Policy Configuration

  e["Solver"]["Discount Factor"] = 0.99;
  e["Solver"]["Mini Batch Size"] = 128;
  e["Solver"]["Critic"]["Learning Rate"] = 0.0001;
  e["Solver"]["Policy"]["Learning Rate"] = 0.000001;
  e["Solver"]["Policy"]["Target Accuracy"] = 0.0001;
  e["Solver"]["Policy"]["Optimization Candidates"] = 8;

  //// Defining Neural Network

  e["Solver"]["Neural Network"]["Engine"] = "OneDNN";

  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32;

  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32;

  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";

  ////// Defining Termination Criteria

  e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 8.0;

  ////// If using syntax test, run for a couple generations only

#ifdef TEST
  e["Solver"]["Termination Criteria"]["Max Generations"] = 20;
#endif

  ////// Setting Korali output configuration

  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Path"] = trainingResultsPath;

  ////// Running Experiment

  auto k = korali::Engine();

  // Configuring profiler output

  k["Profiling"]["Detail"] = "Full";
  k["Profiling"]["Path"] = trainingResultsPath + std::string("/profiling.json");
  k["Profiling"]["Frequency"] = 60;

#ifndef TEST
  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Communicator"] = MPI_COMM_WORLD;
#endif

  k.run(e);

  ////// Now testing policy, dumping trajectory results

#ifndef TEST

  printf("[Korali] Done with training. Now running learned policy to dump the trajectory.\n");

  // Adding custom setting to run the environment dumping the state files during testing
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.1;
  e["Problem"]["Custom Settings"]["Dump Path"] = testingResultsPath;

  e["File Output"]["Path"] = testingResultsPath;
  k["Profiling"]["Path"] = testingResultsPath + std::string("/profiling.json");
  e["Solver"]["Testing"]["Policy"] = e["Solver"]["Best Training Hyperparamters"];
  e["Solver"]["Mode"] = "Testing";
  for (int i = 0; i < N; i++) e["Solver"]["Testing"]["Sample Ids"][i] = i;

  k.run(e);

  printf("[Korali] Finished. Testing dump files stored in %s\n", testingResultsPath.c_str());

#endif
}
