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

  korali["Solver"]["Method"] = "TMCMC";
  korali["Solver"]["Covariance Scaling"] = 0.02;
  korali["Solver"]["Population Size"] = 5000;
  korali["Solver"]["Burn In"] = 5;
  korali["Solver"]["Coefficient of Variation"] = 0.5;

  korali["Seed"] = 0xC0FFEE;
  korali["Verbosity"] = "Detailed";

  korali.run();

 return 0;
}
