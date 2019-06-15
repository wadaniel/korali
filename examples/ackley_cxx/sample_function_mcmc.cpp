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

 k["Solver"]["Method"] = "MCMC";
 k["Solver"]["Population Size"] = 1000;
 k["Solver"]["Burn In"] = 100;
 k["Verbosity"] = "Detailed";

 //k.setModel([](Korali::ModelData& d) { l_gauss(d.getVariables(), d.getResults()); });
 k.setModel([](Korali::ModelData& d) { m_ackley(d.getVariables(), d.getResults()); });

 k.run();

 return 0;
}
