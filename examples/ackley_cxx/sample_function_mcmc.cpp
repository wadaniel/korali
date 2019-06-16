#include "korali.h"
#include "model/ackley.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"] = "Direct";
 k["Solver"]  = "MCMC";

 k["Variables"][0]["Name"] = "X0";
 k["Variables"][1]["Name"] = "X1";
 k["Variables"][2]["Name"] = "X2";
 k["Variables"][3]["Name"] = "X3";

 k["MCMC"]["Population Size"] = 10000;
 k["MCMC"]["Burn In"] = 500;

 k.setModel([](Korali::ModelData& d) { m_ackley(d.getVariables(), d.getResults()); });
 k.run();

 return 0;
}
