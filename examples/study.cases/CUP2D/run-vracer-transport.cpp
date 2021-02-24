// Select which environment to use
#include "_models/transportEnvironment/transportEnvironment.hpp"
#include "korali.hpp"

std::string _resultsPath;

int main(int argc, char *argv[])
{
  // Gathering actual arguments from MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided != MPI_THREAD_FUNNELED)
  {
    printf("Error initializing MPI\n");
    exit(-1);
  }

  // Storing parameters
  _argc = argc;
  _argv = argv;

  // Getting number of workers
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine

  // Init CUP2D
  _environment = new Simulation(_argc, _argv);
  _environment->init();

  std::string trainingResultsPath = "_results_transport_training/";
  std::string testingResultsPath = "_results_transport_testing/";

  // Creating Experiment
  auto e = korali::Experiment();
  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  
  auto found = e.loadState(trainingResultsPath + std::string("/latest"));
  if (found == true) printf("[Korali] Continuing execution from previous run...\n");

  // Configuring Experiment
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Training Reward Threshold"] = 8.0;
  e["Problem"]["Policy Testing Episodes"] = 5;
  e["Problem"]["Actions Between Policy Updates"] = 1;

  // Adding custom setting to run the environment without dumping the state files during training
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.0;
  e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;

  // Setting up the 20 state variables
  const size_t numStates = 20;
  size_t curVariable = 0;
  for (; curVariable < numStates; curVariable++)
  {
    e["Variables"][curVariable]["Name"] = std::string("StateVar") + std::to_string(curVariable);
    e["Variables"][curVariable]["Type"] = "State";
  }

  const double maxForce = 1e-2;

  e["Variables"][curVariable]["Name"] = "Force X";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -maxForce;
  e["Variables"][curVariable]["Upper Bound"] = +maxForce;
  e["Variables"][curVariable]["Initial Exploration Noise"] = 0.5*maxForce;

  curVariable++;
  e["Variables"][curVariable]["Name"] = "Force Y";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -maxForce;
  e["Variables"][curVariable]["Upper Bound"] = +maxForce;
  e["Variables"][curVariable]["Initial Exploration Noise"] = 0.5*maxForce;

  /// Defining Agent Configuration

  e["Solver"]["Type"] = "Agent / Continuous / VRACER";
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Agent Count"] = N;
  e["Solver"]["Episodes Per Generation"] = 1;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Learning Rate"] = 1e-4;
  e["Solver"]["Discount Factor"] = 0.95;

  /// Defining the configuration of replay memory

  e["Solver"]["Experience Replay"]["Start Size"] = 1024;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;

  //// Configuring Mini Batch

  e["Solver"]["Mini Batch Size"] = 128;
  e["Solver"]["Mini Batch Strategy"] = "Uniform";

  //// Defining Neural Network

  // two GRU layers with 32
  // e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
  // e["Solver"]["Time Sequence Length"] = 16;
  // e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Recurrent/GRU";
  // e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32;
  // e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Recurrent/GRU";
  // e["Solver"]["Neural Network"]["Hidden Layers"][1]["Output Channels"] = 32;

  // one GRU layer with 64
  // e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
  // e["Solver"]["Time Sequence Length"] = 16;
  // e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Recurrent/GRU";
  // e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 64;

  // two FF layers with 128
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

  e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 16.0;

  ////// Setting Korali output configuration

  e["Console Output"]["Verbosity"] = "Detailed";
  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Path"] = trainingResultsPath;

  ////// Running Experiment

  auto k = korali::Engine();

  // Configuring profiler output

  k["Profiling"]["Detail"] = "Full";
  k["Profiling"]["Path"] = trainingResultsPath + std::string("/profiling.json");
  k["Profiling"]["Frequency"] = 60;

  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Communicator"] = MPI_COMM_WORLD;

  k.run(e);

  ////// Now testing policy, dumping trajectory results

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
}
