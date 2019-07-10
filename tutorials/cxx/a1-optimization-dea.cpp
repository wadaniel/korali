#include "korali.h"
#include "model/evaluateModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 
 k.setModel([](Korali::ModelData& d) { evaluateModel(d.getVariables(), d.getResults()); });

 k["Problem"] = "Direct Evaluation";
 k["Solver"]  = "DEA";

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["DEA"]["Lower Bound"] = -10.0;
 k["Variables"][0]["DEA"]["Upper Bound"] = +10.0;

 k["DEA"]["Objective"] = "Maximize";
 k["DEA"]["Sample Count"] = 32;
 k["DEA"]["Termination Criteria"]["Max Generations"]["Value"] = 500;

 k["Result Directory"] = "_a1_optimization_dea_result";
 
 k.run();
}
