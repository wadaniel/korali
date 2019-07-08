#include "korali.h"
#include "model/directModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 
 k.setModel([](Korali::ModelData& d) { directModel(d.getVariables(), d.getResults()); });

 k["Problem"] = "Direct Evaluation";
 k["Solver"]  = "DE";

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["DE"]["Lower Bound"] = -10.0;
 k["Variables"][0]["DE"]["Upper Bound"] = +10.0;

 k["DE"]["Objective"] = "Maximize";
 k["DE"]["Sample Count"] = 32;

 k["DE"]["Termination Criteria"]["Max Generations"]["Value"] = 500;

 k["Result Directory"] = "_a1_optimization_de_result";
 
 k.run();
}
