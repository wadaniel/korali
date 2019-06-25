#include "korali.h"
#include "model/directModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"] = "Bayesian";
 k["Solver"] = "TMCMC";

 k["Bayesian"]["Likelihood"]["Type"] = "Direct";
 k.setLikelihood([](Korali::ModelData& d) { directModel(d.getVariables(), d.getResults()); });

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -10.0;
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +10.0;

 k["TMCMC"]["Covariance Scaling"] = 0.02;
 k["TMCMC"]["Population Size"] = 5000;
 k["TMCMC"]["Min Rho Update"] = 0.0;
 k["TMCMC"]["Coefficient of Variation"] = 0.5;
 k["TMCMC"]["Burn In"] = 5;

 k.run();
}
