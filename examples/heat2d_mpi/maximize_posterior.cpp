#include "model/heat2d.hpp"
#include "korali.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 auto p = heat2DInit();

 k["Seed"] = 0xC0FFEE;
 k["Verbosity"] = "Detailed";

 k["Problem"]["Evaluation Type"] = "Bayesian";
 k["Problem"]["Reference Data"] = p.refTemp;

 k["Solver"]["Method"] = "CMA-ES";
 k["Solver"]["Sample Count"] = 32;

 k["Conduit"]["Type"] = "MPI";

 k["Problem"]["Variables"][0]["Name"] = "Intensity";
 k["Problem"]["Variables"][0]["Type"] = "Computational";
 k["Problem"]["Variables"][0]["Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][0]["Distribution"]["Minimum"] = 10.0;
 k["Problem"]["Variables"][0]["Distribution"]["Maximum"] = 60.0;

 k["Problem"]["Variables"][1]["Name"] = "PosX";
 k["Problem"]["Variables"][1]["Type"] = "Computational";
 k["Problem"]["Variables"][1]["Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][1]["Distribution"]["Minimum"] = 0.0;
 k["Problem"]["Variables"][1]["Distribution"]["Maximum"] = 0.5;

 k["Problem"]["Variables"][2]["Name"] = "PosY";
 k["Problem"]["Variables"][2]["Type"] = "Computational";
 k["Problem"]["Variables"][2]["Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][2]["Distribution"]["Minimum"] = 0.6;
 k["Problem"]["Variables"][2]["Distribution"]["Maximum"] = 1.0;

 k["Problem"]["Variables"][3]["Name"] = "Sigma";
 k["Problem"]["Variables"][3]["Type"] = "Statistical";
 k["Problem"]["Variables"][3]["Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][3]["Distribution"]["Minimum"] = 0.0;
 k["Problem"]["Variables"][3]["Distribution"]["Maximum"] = 20.0;

 k["Solver"]["Variables"][0]["Initial Mean"] = 30.0;
 k["Solver"]["Variables"][1]["Initial Mean"] = 0.25;
 k["Solver"]["Variables"][2]["Initial Mean"] = 0.80;
 k["Solver"]["Variables"][3]["Initial Mean"] = 10.0;

 k["Solver"]["Variables"][0]["Initial Standard Deviation"] = 0.300;
 k["Solver"]["Variables"][1]["Initial Standard Deviation"] = 0.001;
 k["Solver"]["Variables"][2]["Initial Standard Deviation"] = 0.001;
 k["Solver"]["Variables"][3]["Initial Standard Deviation"] = 0.100;

 k.setModel([](Korali::ModelData& d) { heat2DSolver(d.getVariables(), d.getResults()); });

 k.run();

 return 0;
}
