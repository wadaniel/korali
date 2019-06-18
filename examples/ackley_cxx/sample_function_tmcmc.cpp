#include "korali.h"
#include "model/ackley.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"] = "Bayesian";
 k["Solver"] = "TMCMC";

 k["Variables"][0]["Name"] = "X0";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -32.0;
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +32.0;

 k["Variables"][1]["Name"] = "X1";
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Minimum"] = -32.0;
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Maximum"] = +32.0;

 k["Variables"][2]["Name"] = "X2";
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Minimum"] = -32.0;
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Maximum"] = +32.0;

 k["Variables"][3]["Name"] = "X3";
 k["Variables"][3]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][3]["Bayesian"]["Prior Distribution"]["Minimum"] = -32.0;
 k["Variables"][3]["Bayesian"]["Prior Distribution"]["Maximum"] = +32.0;

 k["TMCMC"]["Covariance Scaling"] = 0.02;
 k["TMCMC"]["Population Size"] = 10000;
 k["TMCMC"]["Min Rho Update"] = 0.0;

 k["Bayesian"]["Likelihood"]["Type"] = "Direct";
 k.setLikelihood([](Korali::ModelData& d) { m_ackley(d.getVariables(), d.getResults()); });

 k.run();

 return 0;
}
