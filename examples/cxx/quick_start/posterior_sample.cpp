#include "korali.h"

void F( std::vector<double>& par, std::vector<double>& result, std::vector<double> x )
{
 for( size_t i=0; i<x.size(); i++) result.push_back( par[0]*x[i] + par[1] );
}

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();

 korali["Problem"]["Type"] = "Bayesian";
 korali["Problem"]["Variables"][0]["Name"] = "a";
 korali["Problem"]["Variables"][0]["Type"] = "Computational";
 korali["Problem"]["Variables"][0]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][0]["Minimum"] = -5.0;
 korali["Problem"]["Variables"][0]["Maximum"] = +5.0;

 korali["Problem"]["Variables"][1]["Name"] = "b";
 korali["Problem"]["Variables"][1]["Type"] = "Computational";
 korali["Problem"]["Variables"][1]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][1]["Minimum"] = -5.0;
 korali["Problem"]["Variables"][1]["Maximum"] = +5.0;

 korali["Problem"]["Variables"][2]["Name"] = "Sigma";
 korali["Problem"]["Variables"][2]["Type"] = "Statistical";
 korali["Problem"]["Variables"][2]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][2]["Minimum"] = 0.0;
 korali["Problem"]["Variables"][2]["Maximum"] = 10.0;

 std::vector<double> x, y;
 x.push_back(1.0); y.push_back(3.2069);
 x.push_back(2.0); y.push_back(4.1454);
 x.push_back(3.0); y.push_back(4.9393);
 x.push_back(4.0); y.push_back(6.0588);
 x.push_back(5.0); y.push_back(6.8425);

 for (size_t i = 0; i < y.size(); i++)
  korali["Problem"]["Reference Data"][i] = y[i];

 korali["Solver"]["Method"] = "TMCMC";
 korali["Solver"]["Covariance Scaling"] = 0.02;
 korali["Solver"]["Population Size"] = 5000;
 korali["Solver"]["Burn In"] = 5;
 korali["Solver"]["Coefficient of Variation"] = 0.5;

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Detailed";

 korali.run([x](Korali::ModelData& d) { F(d.getParameters(), d.getResults(), x); });

 return 0;
}
