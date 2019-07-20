// In this example, we demonstrate how Korali finds values for the
// variables that maximize the objective function, given by a
// user-provided computational model, subject to a set of
// constraints.

#include "korali.h"
#include "model/g09.h"

int main(int argc, char* argv[])
{
 // Starting Korali's Engine
 auto k = Korali::Engine();

 // Setting model and constraints
 k.setModel([](Korali::Model& d) { g09(d.getVariables(), d.getResults()); });
 k.addConstraint(g1);
 k.addConstraint(g2);
 k.addConstraint(g3);
 k.addConstraint(g4);

 // Selecting problem type
 k["Problem"]["Type"] = "Constrained Optimization";
 k["Problem"]["Objective"] = "Maximize";

 // Setting up 7 variables
 for (int i = 0; i < 7; i++)
 {
  k["Variables"][i]["Name"] = "X" + std::to_string(i);
  k["Variables"][i]["Lower Bound"] = -10.0;
  k["Variables"][i]["Upper Bound"] = +10.0;
 }

 // Configuring the constrained optimizer CCMAES
 k["Solver"]["Type"]  = "CCMAES";
 k["Solver"]["Is Sigma Bounded"] = true;
 k["Solver"]["Sample Count"] = 32;
 k["Solver"]["Viability Sample Count"] = 4;
 k["Solver"]["Termination Criteria"]["Max Fitness"]["Enabled"] = true;
 k["Solver"]["Termination Criteria"]["Max Fitness"]["Value"] = -680.630057374402 - 1e-4;


 // General Settings
 k["General"]["Max Generations"] = 500;
 k["General"]["Console Output"]["Verbosity"] = "Minimal";
 k["General"]["Random Seed"] = 5772;
 k["General"]["Results Output"]["Path"] = "_b2_constrained_optimization_result";

 // Running Korali
 k.run();
}
