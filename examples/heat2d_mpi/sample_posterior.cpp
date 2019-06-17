#include "model/heat2d.hpp"
#include "korali.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 auto p = heat2DInit();

 k["Problem"] = "Bayesian";
 k["Solver"]  = "TMCMC";

 k["Bayesian"]["Likelihood"]["Type"] = "Reference";
 k["Bayesian"]["Likelihood"]["Reference Data"] = p.refTemp;

 k["Variables"][0]["Name"] = "Intensity";
 k["Variables"][0]["Type"] = "Computational";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = 10.0;
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = 60.0;

 k["Variables"][1]["Name"] = "PosX";
 k["Variables"][1]["Type"] = "Computational";
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Maximum"] = 0.5;

 k["Variables"][2]["Name"] = "PosY";
 k["Variables"][2]["Type"] = "Computational";
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Minimum"] = 0.6;
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Maximum"] = 1.0;

 k["Variables"][3]["Name"] = "Sigma";
 k["Variables"][3]["Type"] = "Statistical";
 k["Variables"][3]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][3]["Bayesian"]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][3]["Bayesian"]["Prior Distribution"]["Maximum"] = 20.0;

 k["TMCMC"]["Covariance Scaling"] = 0.02;
 k["TMCMC"]["Population Size"] = 1000;

 k.setModel([](Korali::ModelData& d) { heat2DSolver(d.getVariables(), d.getResults()); });
 k.run();

 return 0;
}
