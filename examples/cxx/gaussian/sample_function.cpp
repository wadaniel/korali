#include "korali.h"
#include "model/gaussian.h"

#define NVARS 4

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine([](Korali::modelData& d) { gaussian(d.getParameters(), d.getResults()); });

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Detailed";

 korali["Problem"]["Type"] = "Direct";
 for (size_t i = 0; i < NVARS; i++)
 {
  korali["Problem"]["Variables"][i]["Name"] = "X" + std::to_string(i);
  korali["Problem"]["Variables"][i]["Type"] = "Computational";
  korali["Problem"]["Variables"][i]["Distribution"] = "Uniform";
  korali["Problem"]["Variables"][i]["Minimum"] = -32.0;
  korali["Problem"]["Variables"][i]["Maximum"] = +32.0;
 }

 korali["Solver"]["Method"] = "TMCMC";
 korali["Solver"]["Covariance Scaling"] = 0.02;
 korali["Solver"]["Population Size"] = 5000;

 gaussian_init(NVARS);
 korali.run();

 return 0;
}
