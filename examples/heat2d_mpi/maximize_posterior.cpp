#include "model/heat2d.hpp"
#include "korali.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 auto p = heat2DInit();

 k["Problem"] = "Bayesian";
 k["Solver"]  = "CMA-ES";
 k["Conduit"] = "MPI";

 k["Likelihood"]["Type"] = "Reference";
 k["Likelihood"]["Reference Data"] = p.refTemp;

 k["Variables"][0]["Name"] = "Intensity";
 k["Variables"][0]["Type"] = "Computational";
 k["Variables"][0]["Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Distribution"]["Minimum"] = 10.0;
 k["Variables"][0]["Distribution"]["Maximum"] = 60.0;
 k["Variables"][0]["CMA-ES"]["Initial Mean"] = 30.0;
 k["Variables"][0]["CMA-ES"]["Initial Standard Deviation"] = 0.300;

 k["Variables"][1]["Name"] = "PosX";
 k["Variables"][1]["Type"] = "Computational";
 k["Variables"][1]["Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Distribution"]["Minimum"] = 0.0;
 k["Variables"][1]["Distribution"]["Maximum"] = 0.5;
 k["Variables"][1]["CMA-ES"]["Initial Mean"] = 0.25;
 k["Variables"][1]["CMA-ES"]["Initial Standard Deviation"] = 0.001;

 k["Variables"][2]["Name"] = "PosY";
 k["Variables"][2]["Type"] = "Computational";
 k["Variables"][2]["Distribution"]["Type"] = "Uniform";
 k["Variables"][2]["Distribution"]["Minimum"] = 0.6;
 k["Variables"][2]["Distribution"]["Maximum"] = 1.0;
 k["Variables"][2]["CMA-ES"]["Initial Mean"] = 0.80;
 k["Variables"][2]["CMA-ES"]["Initial Standard Deviation"] = 0.001;

 k["Variables"][3]["Name"] = "Sigma";
 k["Variables"][3]["Type"] = "Statistical";
 k["Variables"][3]["Distribution"]["Type"] = "Uniform";
 k["Variables"][3]["Distribution"]["Minimum"] = 0.0;
 k["Variables"][3]["Distribution"]["Maximum"] = 20.0;
 k["Variables"][3]["CMA-ES"]["Initial Mean"] = 10.0;
 k["Variables"][3]["CMA-ES"]["Initial Standard Deviation"] = 0.100;

 k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Value"] = 100;
 k["CMA-ES"]["Sample Count"] = 32;

 k.setModel([](Korali::ModelData& d) { heat2DSolver(d.getVariables(), d.getResults()); });
 k.run();

 return 0;
}
