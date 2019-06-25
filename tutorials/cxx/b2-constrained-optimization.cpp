#include "korali.h"
#include "model/g09.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"] = "Direct Evaluation";
 k["Solver"]  = "CMA-ES";

 k.setModel([](Korali::ModelData& d) { g09(d.getVariables(), d.getResults()); });
 k.addConstraint(g1);
 k.addConstraint(g2);
 k.addConstraint(g3);
 k.addConstraint(g4);

 int nParams = 7;
 for (int i = 0; i < nParams; i++)
 {
  k["Variables"][i]["Name"] = "X" + std::to_string(i);
  k["Variables"][0]["CMA-ES"]["Lower Bound"] = -10.0;
  k["Variables"][0]["CMA-ES"]["Upper Bound"] = +10.0;
 }

 k["CMA-ES"]["Adaption Size"] = 0.1;
 k["CMA-ES"]["Sample Count"] = 8;
 k["CMA-ES"]["Viability"]["Sample Count"] = 2;
 k["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"] = -680.630057374402 - 1e-4;

 k.run();
}
