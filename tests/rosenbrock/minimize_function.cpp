#include "korali.h"
#include "model/rosenbrock.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine([](double *x) {return -rosenbrock(x);});

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Normal";

 for (int i = 0; i < NDIMS; i++)
 {
	 korali["Parameters"][i]["Name"] = "X" + std::to_string(i);
	 korali["Parameters"][i]["Distribution"]["Type"] = "Uniform";
	 korali["Parameters"][i]["Distribution"]["Minimum"] = -3.0;
	 korali["Parameters"][i]["Distribution"]["Maximum"] = +3.0;
 }

 korali["Problem"]["Objective"] = "Direct Evaluation";

 korali["Solver"]["Method"] = "CMA-ES";
 korali["Solver"]["Termination Criteria"]["StopMinDeltaX"] = 1e-11;
 korali["Solver"]["Lambda"] = 128;

 korali.run();

 return 0;
}
