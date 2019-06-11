#include "k.h"
#include "model/ackley.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Seed"] = 0xC0FFEE;
 k["Verbosity"] = "Detailed";

 k["Problem"]["Type"] = "Direct";
 for (int i = 0; i < 4; i++) k["Problem"]["Variables"][i]["Name"] = "X" + std::to_string(i);

 k["Solver"]["Method"] = "CMA-ES";
 k["Solver"]["Sample Count"] = 10;
 k["Solver"]["Termination Criteria"]["Max Generations"]["Value"] = 100;
 k["Solver"]["Termination Criteria"]["Min DeltaX"]["Value"] = 1e-7;

 for (int i = 0; i < 4; i++) k["Solver"]["Lower Bounds"][i] = -32.0;
 for (int i = 0; i < 4; i++) k["Solver"]["Upper Bounds"][i] = +32.0;

 k.setModel([](Korali::ModelData& d) { m_ackley(d.getVariables(), d.getResults()); });

 k.run();

 return 0;
}
