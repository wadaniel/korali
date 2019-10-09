#include "model/heat2d.h"
#include "korali.hpp"

int main(int argc, char* argv[])
{
 auto k = korali::Engine();
 auto p = heat2DInit(&argc, &argv);

 k["Problem"]["Type"] = "Evaluation/Bayesian/Inference/Reference";
 k["Problem"]["Likelihood Model"] = "Additive Normal";
 k["Problem"]["Reference Data"] = p.refTemp;
 k["Problem"]["Computational Model"] = &heat2DSolver;

 k["Solver"]["Type"] = "Optimizer/CMAES";
 k["Solver"]["Population Size"] = 32;
 k["Solver"]["Termination Criteria"]["Max Generations"] = 20;

 k["Variables"][0]["Name"] = "Intensity";
 k["Variables"][0]["Bayesian Type"] = "Computational";
 k["Variables"][0]["Prior Distribution"]["Type"] = "Univariate/Uniform";
 k["Variables"][0]["Prior Distribution"]["Minimum"] = 10.0;
 k["Variables"][0]["Prior Distribution"]["Maximum"] = 60.0;
 k["Variables"][0]["Initial Mean"] = 30.0;
 k["Variables"][0]["Initial Standard Deviation"] = 5.0;

 k["Variables"][1]["Name"] = "PosX";
 k["Variables"][1]["Bayesian Type"] = "Computational";
 k["Variables"][1]["Prior Distribution"]["Type"] = "Univariate/Uniform";
 k["Variables"][1]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][1]["Prior Distribution"]["Maximum"] = 0.5;
 k["Variables"][1]["Initial Mean"] = 0.25;
 k["Variables"][1]["Initial Standard Deviation"] = 0.01;

 k["Variables"][2]["Name"] = "PosY";
 k["Variables"][2]["Bayesian Type"] = "Computational";
 k["Variables"][2]["Prior Distribution"]["Type"] = "Univariate/Uniform";
 k["Variables"][2]["Prior Distribution"]["Minimum"] = 0.6;
 k["Variables"][2]["Prior Distribution"]["Maximum"] = 1.0;
 k["Variables"][2]["Initial Mean"] = 0.8;
 k["Variables"][2]["Initial Standard Deviation"] = 0.1;

 k["Variables"][3]["Name"] = "Sigma";
 k["Variables"][3]["Bayesian Type"] = "Statistical";
 k["Variables"][3]["Prior Distribution"]["Type"] = "Univariate/Uniform";
 k["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][3]["Prior Distribution"]["Maximum"] = 20.0;
 k["Variables"][3]["Initial Mean"] = 10.0;
 k["Variables"][3]["Initial Standard Deviation"] = 1.0;

 k.run();

 return 0;
}
