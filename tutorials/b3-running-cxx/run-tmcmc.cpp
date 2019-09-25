#include "model/heat2d.h"
#include "korali.hpp"

int main(int argc, char* argv[])
{
 auto k = korali::Engine();
 auto p = heat2DInit(&argc, &argv);

 k["Problem"]["Type"] = "Evaluation/Bayesian/Inference/Reference";
 k["Problem"]["Likelihood Model"] = "Additive Normal";
 k["Problem"]["Reference Data"] = p.refTemp;
 k["Problem"]["Computational Model"] = [](korali::Sample& d) { heat2DSolver(d.getSampleData(), d.getResult()); };

 k["Solver"]["Type"] = "Sampler/TMCMC";
 k["Solver"]["Population Size"] = 200;

 k["Variables"][0]["Name"] = "Intensity";
 k["Variables"][0]["Bayesian Type"] = "Computational";
 k["Variables"][0]["Prior Distribution"]["Type"] = "Univariate/Uniform";
 k["Variables"][0]["Prior Distribution"]["Minimum"] = 10.0;
 k["Variables"][0]["Prior Distribution"]["Maximum"] = 60.0;

 k["Variables"][1]["Name"] = "PosX";
 k["Variables"][1]["Bayesian Type"] = "Computational";
 k["Variables"][1]["Prior Distribution"]["Type"] = "Univariate/Uniform";
 k["Variables"][1]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][1]["Prior Distribution"]["Maximum"] = 0.5;

 k["Variables"][2]["Name"] = "PosY";
 k["Variables"][2]["Bayesian Type"] = "Computational";
 k["Variables"][2]["Prior Distribution"]["Type"] = "Univariate/Uniform";
 k["Variables"][2]["Prior Distribution"]["Minimum"] = 0.6;
 k["Variables"][2]["Prior Distribution"]["Maximum"] = 1.0;

 k["Variables"][3]["Name"] = "Sigma";
 k["Variables"][3]["Bayesian Type"] = "Statistical";
 k["Variables"][3]["Prior Distribution"]["Type"] = "Univariate/Uniform";
 k["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][3]["Prior Distribution"]["Maximum"] = 20.0;

 k.run();

 return 0;
}
