#include "korali.h"
#include "model/directModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 k.setLikelihood([](Korali::ModelData& d) { directModel(d.getVariables(), d.getResults()); });

 k["Problem"] = "Bayesian";
 k["Solver"] = "TMCMC";

 k["Bayesian"]["Likelihood"]["Type"] = "Direct";

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -10.0;
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +10.0;

 k["TMCMC"]["Population Size"] = 5000;

 k["Result Directory"] = "_a2_sampling_tmcmc";
 
 k.run();
}
