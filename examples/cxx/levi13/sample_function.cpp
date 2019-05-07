#include "korali.h"
#include "model/levi13.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine([](Korali::modelData& d) {
	 eggholder(d.getParameters(), d.getResults());
 });

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Normal";

 for (int i = 0; i < 2; i++)
 {
  korali["Parameters"][i]["Name"] = "X" + std::to_string(i);
  korali["Parameters"][i]["Distribution"] = "Uniform";
  korali["Parameters"][i]["Type"] = "Computational";
  korali["Parameters"][i]["Minimum"] = -10.0;
  korali["Parameters"][i]["Maximum"] = +10.0;
 }

 korali["Problem"]["Objective"] = "Direct Evaluation";

 korali["Solver"]["Method"] = "TMCMC";
 korali["Solver"]["Covariance Scaling"] = 0.04;
 korali["Solver"]["Population Size"] = 30000;

 korali.run();

 return 0;
}
