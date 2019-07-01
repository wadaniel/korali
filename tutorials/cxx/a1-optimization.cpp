#include "korali.h"
#include "model/directModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"] = "Direct Evaluation";
 k["Solver"]  = "CMA-ES";

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["CMA-ES"]["Lower Bound"] = -10.0;
 k["Variables"][0]["CMA-ES"]["Upper Bound"] = +10.0;

 k["CMA-ES"]["Objective"] = "Maximize";
 k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Value"] = 500;
 k["CMA-ES"]["Sample Count"] = 5;

 k.setModel([](Korali::ModelData& d) { directModel(d.getVariables(), d.getResults()); });
 auto tmp = k.getConfiguration();
 k.run();
}
