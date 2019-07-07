#include "korali.h"
#include "model/directModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"] = "Direct Evaluation";
 k["Solver"]  = "CMAES";

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["CMAES"]["Lower Bound"] = -10.0;
 k["Variables"][0]["CMAES"]["Upper Bound"] = +10.0;

 k["CMAES"]["Objective"] = "Maximize";
 k["CMAES"]["Sample Count"] = 32;

 k["Termination Criteria"]["Max Generations"]["Value"] = 500;
 k.setModel([](Korali::ModelData& d) { directModel(d.getVariables(), d.getResults()); });
 k.run();
}
