#include "korali.h"
#include "model/directModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"] = "Direct Evaluation";
 k["Solver"] = "MCMC";

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["MCMC"]["Initial Mean"] = 0.0;
 k["Variables"][0]["MCMC"]["Standard Deviation"] = 1.000;

 k["MCMC"]["Burn In"] = 500;
 k["MCMC"]["Chain Length"]  = 5000;
 k["MCMC"]["Use Adaptive Sampling"]  = true;
 k["MCMC"]["Result Output Frequency"]  = 5000;

 k.setModel([](Korali::ModelData& d) { directModel(d.getVariables(), d.getResults()); });

 k.run();
}
