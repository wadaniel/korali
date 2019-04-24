#include "korali.h"
#include "model/ackley.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine([](Korali::modelData& d) {
	 ackley(d.getParameters(), d.getResults());
 });

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Detailed";

 for (int i = 0; i < 4; i++)
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
 korali["Solver"]["Burn In"] = 5;

 korali.run();

 return 0;
}
