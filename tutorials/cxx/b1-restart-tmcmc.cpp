#include "korali.h"
#include "model/evaluateModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 k.setLikelihood([](Korali::ModelData& d) { evaluateModel(d.getVariables(), d.getResults()); });

 k["Problem"] = "Bayesian";
 k["Solver"] = "TMCMC";

 k["Bayesian"]["Likelihood"]["Type"] = "Direct";

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -10.0;
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +10.0;

 k["TMCMC"]["Population Size"] = 5000;

 k["Result Directory"] = "_b1_restart_tmcmc";

 k.run();

 printf("\n\nRestarting now:\n\n");

 k.loadState("_b1_restart_tmcmc/s00003.json");

 k.run();
}
