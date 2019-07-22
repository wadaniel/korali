// In this example, we demonstrate how Korali finds values for the
// variables that maximize the objective function, given by a
// user-provided computational model.

#include "korali.h"
#include "model/ellipsoid.h"

int main(int argc, char* argv[])
{
  // Starting Korali's Engine
  auto k = Korali::Engine();

  // Setting computational model
  k.setModel([](Korali::Model& d) { ellipsoid(d.getVariables(), d.getResults()); });

  // Selecting problem and solver types.
  k["Problem"] = "Direct Evaluation";
  k["Solver"]  = "CMAES";

  // Defining the problem's variables and their CMA-ES bounds.
  size_t nparams = 10;
  for(size_t i = 0; i < nparams; ++i)
  {
    k["Variables"][i]["Name"] = "X" + std::to_string(i);
    k["Variables"][i]["CMAES"]["Initial Mean"] = 1.0;
    k["Variables"][i]["CMAES"]["Lower Bound"]  = -20.0;
    k["Variables"][i]["CMAES"]["Upper Bound"]  = +20.0;

    /*
    if( i == 0 || i == 1 || i == 3 || i == 6)
    {
      k["Variables"][i]["CMAES"]["Discrete"]     = true;
      k["Variables"][i]["CMAES"]["Granularity"]  = 1.0;
    }
    */
  }

  // Configuring CMA-ES parameters
  k["CMAES"]["Objective"]    = "Maximize";
  k["CMAES"]["Sample Count"] = 100;

  k["CMAES"]["Termination Criteria"]["Max Generations"]["Value"] = 1000;

  // Setting output directory
  k["Result Directory"] = "_b3_integer_optimization_result";

  // Running Korali
  k.run();

}
