#include "model/heat2d.h"
#include "korali.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 auto p = heat2DInit(&argc, &argv);

 int teamSize = 1;
 if (argc == 1) teamSize = atoi(argv[0]);

 k["Problem"] = "Bayesian";
 k["Bayesian"]["Likelihood"]["Type"] = "Reference";
 k["Bayesian"]["Likelihood"]["Model"] = "Additive Gaussian";
 k["Bayesian"]["Likelihood"]["Reference Data"] = p.refTemp;

 k["Variables"][0]["Name"] = "Intensity";
 k["Variables"][0]["Bayesian"]["Type"] = "Computational";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = 10.0;
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = 60.0;
 k["Variables"][0]["CMAES"]["Lower Bound"] = 10.0;
 k["Variables"][0]["CMAES"]["Upper Bound"] = 60.0;

 k["Variables"][1]["Name"] = "PosX";
 k["Variables"][1]["Bayesian"]["Type"] = "Computational";
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Maximum"] = 0.5;
 k["Variables"][1]["CMAES"]["Lower Bound"] = 0.0;
 k["Variables"][1]["CMAES"]["Upper Bound"] = 0.5;

 k["Variables"][2]["Name"] = "PosY";
 k["Variables"][2]["Bayesian"]["Type"] = "Computational";
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Minimum"] = 0.6;
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Maximum"] = 1.0;
 k["Variables"][2]["CMAES"]["Lower Bound"] = 0.6;
 k["Variables"][2]["CMAES"]["Upper Bound"] = 1.0;

 k["Variables"][3]["Name"] = "Sigma";
 k["Variables"][3]["Bayesian"]["Type"] = "Statistical";
 k["Variables"][3]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][3]["Bayesian"]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][3]["Bayesian"]["Prior Distribution"]["Maximum"] = 20.0;
 k["Variables"][3]["CMAES"]["Lower Bound"] = 0.0;
 k["Variables"][3]["CMAES"]["Upper Bound"] = 20.0;

 k["Conduit"]= "Distributed";
 k["Distributed"]["Ranks Per Team"] = teamSize;

 k["Solver"] = "CMAES";
 k["CMAES"]["Sample Count"] = 32;

 k.setModel([](Korali::ModelData& d) { heat2DSolver(d.getVariables(), d.getResults()); });

 k.run();

 return 0;
}
