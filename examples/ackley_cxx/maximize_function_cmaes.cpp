#include "korali.h"
#include "model/ackley.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"] = "Direct";
 k["Solver"]  = "CMA-ES";

 k["Variables"][0]["Name"] = "X0";
 k["Variables"][0]["CMA-ES"]["Lower Bound"] = -32.0;
 k["Variables"][0]["CMA-ES"]["Upper Bound"] = +32.0;

 k["Variables"][1]["Name"] = "X1";
 k["Variables"][1]["CMA-ES"]["Lower Bound"] = -32.0;
 k["Variables"][1]["CMA-ES"]["Upper Bound"] = +32.0;

 k["Variables"][2]["Name"] = "X2";
 k["Variables"][2]["CMA-ES"]["Lower Bound"] = -32.0;
 k["Variables"][2]["CMA-ES"]["Upper Bound"] = +32.0;

 k["Variables"][3]["Name"] = "X3";
 k["Variables"][3]["CMA-ES"]["Lower Bound"] = -32.0;
 k["Variables"][3]["CMA-ES"]["Upper Bound"] = +32.0;

 k["CMA-ES"]["Objective"] = "Maximize";
 k["CMA-ES"]["Sample Count"] = 10;
 k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Value"] = 100;

 k.setModel([](Korali::ModelData& d) { m_ackley(d.getVariables(), d.getResults()); });

 k.run();

 return 0;
}
