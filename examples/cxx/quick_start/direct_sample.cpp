#include "korali.h"

void F(std::vector<double>& x, std::vector<double>& result)
{
 result.push_back( -pow( x[0]-2. , 2) );
}

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();

 korali["Problem"]["Type"] = "Direct";

 korali["Problem"]["Variables"][0]["Name"] = "X0";
 korali["Problem"]["Variables"][0]["Type"] = "Computational";
 korali["Problem"]["Variables"][0]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][0]["Minimum"] = -10.0;
 korali["Problem"]["Variables"][0]["Maximum"] = +10.0;

 korali["Solver"]["Method"] = "TMCMC";
 korali["Solver"]["Covariance Scaling"] = 0.02;
 korali["Solver"]["Population Size"] = 5000;
 korali["Solver"]["Burn In"] = 5;
 korali["Solver"]["Coefficient of Variation"] = 0.5;

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Detailed";

 korali.run([](Korali::ModelData& d) { F(d.getParameters(), d.getResults()); });
 return 0;
}
