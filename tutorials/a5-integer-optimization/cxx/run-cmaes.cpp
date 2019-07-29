// In this example, we demonstrate how Korali finds values for the
// variables that maximize the objective function, given by a
// user-provided mixed integer computational model.

#include "korali.hpp"
#include "model/model.h"

int main(int argc, char* argv[])
{
  // Starting Korali's Engine
  auto k = Korali::Engine();

  // Setting computational model
  k.setModel(model);

  // Selecting problem and solver types.
  k["Problem"]["Type"] = "Optimization";
  k["Problem"]["Objective"] = "Maximize";

  // Defining the problem's variables and their CMA-ES bounds.
  size_t nparams = 10;
  for(size_t i = 0; i < nparams; ++i)
  {
    k["Variables"][i]["Name"] = "X" + std::to_string(i);
    k["Variables"][i]["Initial Mean"] = 1.0;
    k["Variables"][i]["Lower Bound"]  = -19.0;
    k["Variables"][i]["Upper Bound"]  = +21.0;

    if( i == 0 || i == 1 || i == 3 || i == 6)
    {
      k["Variables"][i]["Is Discrete"]    = true;
      k["Variables"][i]["Granularity"] = 1.0;
    }
  }

  // Configuring CMA-ES parameters
  k["Solver"]["Type"] = "CMAES";
  k["Solver"]["Sample Count"] = 8;
  k["Solver"]["Termination Criteria"]["Min Fitness Diff Threshold"] = 1e-9;
  k["Solver"]["Termination Criteria"]["Max Generations"] = 500;

  k["General"]["Console Output"]["Frequency"] = 50;
  k["General"]["Results Output"]["Frequency"] = 50;

  // Running Korali
  k.run();

}
