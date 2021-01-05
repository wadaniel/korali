#include "_environment/environment.hpp"
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

  // Initializing Cubism
  _environment = new Simulation(_argc, _argv);
  _environment->init();

  // Setting results path
  std::string trainingResultsPath = "_trainingResults";
  std::string testingResultsPath = "_testingResults";

  // Creating Experiment
  auto e = korali::Experiment();
  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";

  ////// Checking if existing results are there and continuing them

  auto found = e.loadState(trainingResultsPath + std::string("/latest"));
  if (found == true) printf("Continuing execution from previous run...\n");

  // Configuring Experiment
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Training Reward Threshold"] = 8.0;
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
  e["Variables"][curVariable]["Exploration Sigma"]["Initial"] = 0.4;
  e["Variables"][curVariable]["Exploration Sigma"]["Final"] = 0.05;
  e["Variables"][curVariable]["Exploration Sigma"]["Annealing Rate"] = 1e-5;

  curVariable++;
  e["Variables"][curVariable]["Name"] = "Force";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -0.25;
  e["Variables"][curVariable]["Upper Bound"] = +0.25;
  e["Variables"][curVariable]["Exploration Sigma"]["Initial"] = 0.1;
  e["Variables"][curVariable]["Exploration Sigma"]["Final"] = 0.02;
  e["Variables"][curVariable]["Exploration Sigma"]["Annealing Rate"] = 1e-5;

  /// Defining Agent Configuration

  e["Solver"]["Type"] = "Agent / Continuous / GFPT";
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Agent Count"] = N;
  e["Solver"]["Episodes Per Generation"] = 1;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Cache Persistence"] = 100;
  e["Solver"]["Learning Rate"] = 0.001;

  /// Defining the configuration of replay memory

  e["Solver"]["Experience Replay"]["Start Size"] = 1024;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;

  /// Configuring the Remember-and-Forget Experience Replay algorithm

  e["Solver"]["Experience Replay"]["REFER"]["Enabled"] = true;
  e["Solver"]["Experience Replay"]["REFER"]["Cutoff Scale"] = 4.0;
  e["Solver"]["Experience Replay"]["REFER"]["Target"] = 0.1;
  e["Solver"]["Experience Replay"]["REFER"]["Initial Beta"] = 0.6;
  e["Solver"]["Experience Replay"]["REFER"]["Annealing Rate"] = 5e-7;

  //// Configuring Mini Batch

  e["Solver"]["Mini Batch Size"] = 128;
  e["Solver"]["Mini Batch Strategy"] = "Uniform";

  //// Defining Critic and Policy Configuration

  e["Solver"]["Critic"]["Advantage Function Population"] = 12;
  e["Solver"]["Policy"]["Learning Rate Scale"] = 0.1;
  e["Solver"]["Policy"]["Target Accuracy"] = 0.01;
  e["Solver"]["Policy"]["Optimization Candidates"] = 24;

  //// Defining Neural Network

  e["Solver"]["Neural Network"]["Engine"] = "OneDNN";

  e["Solver"]["Time Sequence Length"] = 16;

  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Recurrent/GRU";
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32;

  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Recurrent/GRU";
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Output Channels"] = 32;

  ////// Defining Termination Criteria

  e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 16.0;

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
