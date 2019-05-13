#include "korali.h"

void F(std::vector<double>& x, std::vector<double>& result){

 result.push_back( -pow( x[0]-2. , 2) );
}

int main(int argc, char* argv[])
{

 auto korali = Korali::Engine([](Korali::modelData& d) {
  F(d.getParameters(), d.getResults());
 });

 korali["Problem"]["Type"] = "Direct";

 korali["Problem"]["Variables"][0]["Name"] = "X0";
 korali["Problem"]["Variables"][0]["Type"] = "Computational";
 korali["Problem"]["Variables"][0]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][0]["Minimum"] = -10.0;
 korali["Problem"]["Variables"][0]["Maximum"] = +10.0;

 korali["Solver"]["Method"] = "CMA-ES";
 korali["Solver"]["Lambda"] = 12;
 korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-11;
 korali["Solver"]["Termination Criteria"]["Min Fitness"] = 1e-12;
 korali["Solver"]["Termination Criteria"]["Max Generations"] = 1e4;
 korali["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 1e4;

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Detailed";

 korali.run();
 return 0;
}
