// Select which environment to use
#include "_transportEnvironment/transportEnvironment.hpp"
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

 
  // Defining constants
  const double maxForce = 1e-2;
  const size_t numVariables = 16;
 
  // Init CUP2D
  _environment = new Simulation(_argc, _argv);
  _environment->init();

  std::string resultsPath = "true_results_transport_mocmaes/";

  // Creating Experiment
  auto e = korali::Experiment();

  auto found = e.loadState(resultsPath + std::string("latest"));
  if (found == true) printf("[Korali] Continuing execution from previous run...\n");

  e["Random Seed"] = 0xC0FEE;
  e["Problem"]["Type"] = "Optimization";
  e["Problem"]["Objective Function"] = &runEnvironmentMocmaes;
  e["Problem"]["Num Objectives"] = 2;
  
 
  // Configuring MO-CMA-ES parameters
  e["Solver"]["Type"] = "Optimizer/MOCMAES";
  e["Solver"]["Population Size"] = 64;
  e["Solver"]["Mu Value"] = 32;
  e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-16;
  e["Solver"]["Termination Criteria"]["Min Variable Difference Threshold"] = 1e-16;
  e["Solver"]["Termination Criteria"]["Max Generations"] = 50;
 
  // Setting up the variables
  for (size_t var = 0; var < numVariables; ++var)
  {
    e["Variables"][var]["Name"] = std::string("Velocity") + std::to_string(var);
    e["Variables"][var]["Lower Bound"] = 0.0;
    e["Variables"][var]["Upper Bound"] = +maxForce;
    e["Variables"][var]["Initial Standard Deviation"] = 0.2*maxForce/std::sqrt(numVariables);
  }

  ////// Setting Korali output configuration

  e["Console Output"]["Verbosity"] = "Detailed";
  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Path"] = resultsPath;

  ////// Running Experiment

  auto k = korali::Engine();

  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Communicator"] = MPI_COMM_WORLD;
  k["Conduit"]["Ranks Per Worker"] = 1;

  k.run(e);
}
