#include "model/heat2d.h"
#include "korali.hpp"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 auto p = heat2DInit(&argc, &argv);

 k["Problem"]["Type"] = "Bayesian Inference";
 k["Problem"]["Likelihood"]["Model"] = "Additive Normal";
 k["Problem"]["Likelihood"]["Reference Data"] = p.refTemp;

 k["Problem"]["Variables"][0]["Name"] = "Intensity";
 k["Problem"]["Variables"][0]["Type"] = "Computational";
 k["Problem"]["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"] = 10.0;
 k["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"] = 60.0;
 k["Problem"]["Variables"][0]["Lower Bound"] = 10.0;
 k["Problem"]["Variables"][0]["Upper Bound"] = 60.0;

 k["Problem"]["Variables"][1]["Name"] = "PosX";
 k["Problem"]["Variables"][1]["Type"] = "Computational";
 k["Problem"]["Variables"][1]["Prior Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][1]["Prior Distribution"]["Minimum"] = 0.0;
 k["Problem"]["Variables"][1]["Prior Distribution"]["Maximum"] = 0.5;
 k["Problem"]["Variables"][1]["Lower Bound"] = 0.0;
 k["Problem"]["Variables"][1]["Upper Bound"] = 0.5;

 k["Problem"]["Variables"][2]["Name"] = "PosY";
 k["Problem"]["Variables"][2]["Type"] = "Computational";
 k["Problem"]["Variables"][2]["Prior Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][2]["Prior Distribution"]["Minimum"] = 0.6;
 k["Problem"]["Variables"][2]["Prior Distribution"]["Maximum"] = 1.0;
 k["Problem"]["Variables"][2]["Lower Bound"] = 0.6;
 k["Problem"]["Variables"][2]["Upper Bound"] = 1.0;

 k["Problem"]["Variables"][3]["Name"] = "Sigma";
 k["Problem"]["Variables"][3]["Type"] = "Statistical";
 k["Problem"]["Variables"][3]["Prior Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0;
 k["Problem"]["Variables"][3]["Prior Distribution"]["Maximum"] = 20.0;
 k["Problem"]["Variables"][3]["Lower Bound"] = 0.0;
 k["Problem"]["Variables"][3]["Upper Bound"] = 20.0;

 k["Solver"]["Type"] = "CMAES";
 k["Solver"]["Population Size"] = 32;
 k["Solver"]["Termination Criteria"]["Max Generations"] = 20;

 k.setReferenceModel([](Korali::Model::Reference& d) { heat2DSolver(d.getVariables(), d.getEvaluations()); });

 k.run();

 return 0;
}
