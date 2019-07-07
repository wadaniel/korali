#include "korali.h"
#include "model/posteriorModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"] = "Bayesian";

 std::vector<double> x, y; // Reference Data
 getReferenceData(x, y);

 k["Bayesian"]["Likelihood"]["Type"] = "Reference";
 k["Bayesian"]["Likelihood"]["Reference Data"] = y;

 k.setModel([x](Korali::ModelData& d) { posteriorModel(d.getVariables(), d.getResults(), x); });

 k["Variables"][0]["Name"] = "a";
 k["Variables"][0]["Bayesian"]["Type"] = "Computational";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0;
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

 k["Variables"][1]["Name"] = "b";
 k["Variables"][1]["Bayesian"]["Type"] = "Computational";
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0;
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

 k["Variables"][2]["Name"] = "Sigma";
 k["Variables"][2]["Bayesian"]["Type"] = "Statistical";
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

 k["Solver"] = "TMCMC";

 k["TMCMC"]["Covariance Scaling"] = 0.02;
 k["TMCMC"]["Population Size"] = 5000;
 k["TMCMC"]["Min Rho Update"] = 0.0;
 k["TMCMC"]["Initial Coefficient of Variation"] = 0.5;
 k["TMCMC"]["Burn In"] = 5;

 k.run();
}
