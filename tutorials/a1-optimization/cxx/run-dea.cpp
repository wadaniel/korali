// In this example, we demonstrate how Korali finds values for the
// variables that maximize the objective function, given by a
// user-provided computational model.

#include "korali.hpp"
#include "model/model.h"

int main(int argc, char* argv[])
{
 // Starting Korali's Engine
 auto k = Korali::Engine();

 // Selecting problem and solver types.
 k["Problem"]["Type"] = "Optimization";
 k["Problem"]["Objective"] = "Maximize";

 // Defining the problem's variables and their CMA-ES bounds.
 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["Lower Bound"] = -10.0;
 k["Variables"][0]["Upper Bound"] = +10.0;

 // Configuring CMA-ES parameters
 k["Solver"]["Type"] = "DEA";
 k["Solver"]["Sample Count"] = 32;
 k["Solver"]["Termination Criteria"]["Min Fitness Diff Threshold"] = 1e-7;
 k["Solver"]["Termination Criteria"]["Max Generations"] = 100;

 // Setting computational model
 k.setModel(model);

 // Running Korali
 k.run();
}
