#include "model/heat2d.h"
#include "korali.hpp"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 auto p = heat2DInit(&argc, &argv);

 k["Problem"]["Type"] = "Bayesian Inference";
 k["Problem"]["Likelihood"]["Model"] = "Additive Gaussian";
 k["Problem"]["Likelihood"]["Reference Data"] = p.refTemp;

 k["Variables"][0]["Name"] = "Intensity";
 k["Variables"][0]["Type"] = "Computational";
 k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Prior Distribution"]["Minimum"] = 10.0;
 k["Variables"][0]["Prior Distribution"]["Maximum"] = 60.0;
 k["Variables"][0]["Lower Bound"] = 10.0;
 k["Variables"][0]["Upper Bound"] = 60.0;

 k["Variables"][1]["Name"] = "PosX";
 k["Variables"][1]["Type"] = "Computational";
 k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][1]["Prior Distribution"]["Maximum"] = 0.5;
 k["Variables"][1]["Lower Bound"] = 0.0;
 k["Variables"][1]["Upper Bound"] = 0.5;

 k["Variables"][2]["Name"] = "PosY";
 k["Variables"][2]["Type"] = "Computational";
 k["Variables"][2]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][2]["Prior Distribution"]["Minimum"] = 0.6;
 k["Variables"][2]["Prior Distribution"]["Maximum"] = 1.0;
 k["Variables"][2]["Lower Bound"] = 0.6;
 k["Variables"][2]["Upper Bound"] = 1.0;

 k["Variables"][3]["Name"] = "Sigma";
 k["Variables"][3]["Type"] = "Statistical";
 k["Variables"][3]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][3]["Prior Distribution"]["Maximum"] = 20.0;
 k["Variables"][3]["Lower Bound"] = 0.0;
 k["Variables"][3]["Upper Bound"] = 20.0;

 k["Solver"]["Type"] = "CMAES";
 k["Solver"]["Sample Count"] = 32;

 k["Conduit"]["Type"] = "MPI";
 k["Conduit"]["Ranks Per Team"] = 1;

 k["General"]["Max Generations"] = 50;

 k.setModel([](Korali::Model& d) { heat2DSolver(d.getVariables(), d.getResults()); });

 k.run();

 return 0;
}
