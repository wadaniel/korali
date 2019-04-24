#include "model/heat2d.hpp"
#include "korali.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine([](Korali::modelData& d) {
	 heat2DSolver (d.getParameters(), d.getResults());
 });

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Normal";

 korali["Parameters"][0]["Name"] = "Intensity";
 korali["Parameters"][0]["Type"] = "Computational";
 korali["Parameters"][0]["Distribution"] = "Uniform";
 korali["Parameters"][0]["Minimum"] = 10.0;
 korali["Parameters"][0]["Maximum"] = 60.0;

 korali["Parameters"][1]["Name"] = "PosX";
 korali["Parameters"][1]["Type"] = "Computational";
 korali["Parameters"][1]["Distribution"] = "Uniform";
 korali["Parameters"][1]["Minimum"] = 0.0;
 korali["Parameters"][1]["Maximum"] = 0.5;

 korali["Parameters"][2]["Name"] = "PosY";
 korali["Parameters"][2]["Type"] = "Computational";
 korali["Parameters"][2]["Distribution"] = "Uniform";
 korali["Parameters"][2]["Minimum"] = 0.6;
 korali["Parameters"][2]["Maximum"] = 1.0;

 korali["Parameters"][3]["Name"] = "Sigma";
 korali["Parameters"][3]["Type"] = "Statistical";
 korali["Parameters"][3]["Distribution"] = "Uniform";
 korali["Parameters"][3]["Minimum"] = 0.0;
 korali["Parameters"][3]["Maximum"] = 20.0;

 korali["Problem"]["Objective"] = "Posterior";

 auto p = heat2DInit();
 for (size_t i = 0; i < p.refTemp.size(); i++)
	 korali["Problem"]["Reference Data"][i] = p.refTemp[i];

 korali["Solver"]["Method"] = "CMA-ES";
 korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-7;
 korali["Solver"]["Lambda"] = 32;

 korali.run();

 return 0;
}
