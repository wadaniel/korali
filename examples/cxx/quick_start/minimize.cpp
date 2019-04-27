#include "korali.h"


void F01(std::vector<double>& x, std::vector<double>& result){

  size_t dim = x.size();

  result.push_back( -pow( x[0]-2. , 2) );
}


int main(int argc, char* argv[])
{
  auto korali = Korali::Engine([](Korali::modelData& d) {
    F01(d.getParameters(), d.getResults());
  });

  korali["Problem"]["Objective"] = "Direct Evaluation";

  korali["Parameters"][0]["Name"] = "X0";
  korali["Parameters"][0]["Type"] = "Computational";
  korali["Parameters"][0]["Distribution"] = "Uniform";
  korali["Parameters"][0]["Minimum"] = -10.0;
  korali["Parameters"][0]["Maximum"] = +10.0;

  korali["Solver"]["Method"] = "CMA-ES";
  korali["Solver"]["Lambda"] = 12;
  korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-11;
  korali["Solver"]["Termination Criteria"]["Min Fitness"] = 1e-12;
  korali["Solver"]["Termination Criteria"]["Max Generations"] = 1e4;
  korali["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 1e4;

  korali["Seed"] = 0xC0FFEE;
  korali["Verbosity"] = "Normal";

  korali.run();
  return 0;
}
