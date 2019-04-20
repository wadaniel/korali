#include "korali.h"
#include "model/gaussian.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine([](Korali::modelData& d) {
	 gaussian(d.getParameters(), d.getResults());
 });

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Normal";

 size_t nPars = 4;
 for (size_t i = 0; i < nPars; i++)
 {
  korali["Parameters"][i]["Name"] = "X" + std::to_string(i);
  korali["Parameters"][i]["Type"] = "Computational";
  korali["Parameters"][i]["Distribution"] = "Uniform";
  korali["Parameters"][i]["Minimum"] = -32.0;
  korali["Parameters"][i]["Maximum"] = +32.0;
 }

 korali["Problem"]["Objective"] = "Direct Evaluation";
 korali["Solver"]["Method"] = "TMCMC";
 korali["Solver"]["Covariance Scaling"] = 0.02;
 korali["Solver"]["Population Size"] = 5000;

 gaussian_init(nPars);
 korali.run();

 return 0;
}
