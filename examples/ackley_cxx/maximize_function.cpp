#include "korali.h"
#include "model/ackley.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"]["Evaluation Type"] = "Direct";

 k["Problem"]["Variables"][0]["Name"] = "X0";
 k["Problem"]["Variables"][1]["Name"] = "X1";
 k["Problem"]["Variables"][2]["Name"] = "X2";
 k["Problem"]["Variables"][3]["Name"] = "X3";

 k["Solver"]["Method"] = "CMA-ES";

 k["Solver"]["Variables"][0]["Lower Bound"] = -32.0;
 k["Solver"]["Variables"][1]["Lower Bound"] = -32.0;
 k["Solver"]["Variables"][2]["Lower Bound"] = -32.0;
 k["Solver"]["Variables"][3]["Lower Bound"] = -32.0;

 k["Solver"]["Variables"][0]["Upper Bound"] = +32.0;
 k["Solver"]["Variables"][1]["Upper Bound"] = +32.0;
 k["Solver"]["Variables"][2]["Upper Bound"] = +32.0;
 k["Solver"]["Variables"][3]["Upper Bound"] = +32.0;

 k["Solver"]["Sample Count"] = 10;
 k["Solver"]["Termination Criteria"]["Max Generations"]["Value"] = 100;
 k["Solver"]["Termination Criteria"]["Min DeltaX"]["Value"] = 1e-7;

 k.setModel([](Korali::ModelData& d) { m_ackley(d.getVariables(), d.getResults()); });

 k.run();

 return 0;
}
