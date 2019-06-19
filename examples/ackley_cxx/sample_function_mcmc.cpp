#include "korali.h"
#include "model/ackley.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"] = "Direct Evaluation";
 k["Solver"]  = "MCMC";
 k["Verbosity"]  = "Detailed";

 k["Variables"][0]["Name"] = "X0";
 k["Variables"][0]["MCMC"]["Initial Mean"] = 0.0;
 k["Variables"][0]["MCMC"]["Initial Standard Deviation"] = 1.000;

 k["Variables"][1]["Name"] = "X1";
 k["Variables"][1]["MCMC"]["Initial Mean"] = 0.0;
 k["Variables"][1]["MCMC"]["Initial Standard Deviation"] = 1.000;

 k["Variables"][2]["Name"] = "X2";
 k["Variables"][2]["MCMC"]["Initial Mean"] = 0.0;
 k["Variables"][2]["MCMC"]["Initial Standard Deviation"] = 1.000;

 k["Variables"][3]["Name"] = "X3";
 k["Variables"][3]["MCMC"]["Initial Mean"] = 0.0;
 k["Variables"][3]["MCMC"]["Initial Standard Deviation"] = 1.000;

 k["MCMC"]["Population Size"] = 5000;
 k["MCMC"]["Burn In"] = 500;
 k["MCMC"]["Adaptive Sampling"]  = true;

 k.setModel([](Korali::ModelData& d) { m_ackley(d.getVariables(), d.getResults()); });
 k.run();

 return 0;
}
