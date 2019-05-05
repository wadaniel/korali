#include "korali.h"
#include <stdlib.h>

void F( std::vector<double>& par, std::vector<double>& result, std::vector<double> x ){
  for( size_t i=0; i<x.size(); i++){
    double res = par[0]*x[i] + par[1];
    result.push_back( res );
  }

}

int main(int argc, char* argv[])
{
  if(argc<2){
    printf("\nUsage: ./posterior 1(=maximize likelihood) or 2(=sample posterior)\n\n");
    exit(1);
  }

  std::vector<double> x, y;
  x.push_back(1.0);   y.push_back(3.2069);
  x.push_back(2.0);   y.push_back(4.1454);
  x.push_back(3.0);   y.push_back(4.9393);
  x.push_back(4.0);   y.push_back(6.0588);
  x.push_back(5.0);   y.push_back(6.8425);

  auto Fx = [x]( Korali::modelData& d ) {
                F(d.getParameters(), d.getResults(), x);
  };

  auto korali = Korali::Engine( Fx );

  for (size_t i = 0; i < y.size(); i++)
    korali["Problem"]["Reference Data"][i] = y[i];


  korali["Parameters"][0]["Name"] = "a";
  korali["Parameters"][0]["Type"] = "Computational";
  korali["Parameters"][0]["Distribution"] = "Uniform";
  korali["Parameters"][0]["Minimum"] = -5.0;
  korali["Parameters"][0]["Maximum"] = +5.0;
  korali["Parameters"][0]["Initial Mean"] = +1.0;

  korali["Parameters"][1]["Name"] = "b";
  korali["Parameters"][1]["Type"] = "Computational";
  korali["Parameters"][1]["Distribution"] = "Uniform";
  korali["Parameters"][1]["Minimum"] = -5.0;
  korali["Parameters"][1]["Maximum"] = +5.0;
  korali["Parameters"][1]["Initial Mean"] = +1.0;

  korali["Parameters"][2]["Name"] = "Sigma";
  korali["Parameters"][2]["Type"] = "Statistical";
  korali["Parameters"][2]["Distribution"] = "Uniform";
  korali["Parameters"][2]["Minimum"] = 0.0;
  korali["Parameters"][2]["Maximum"] = 10.0;
  korali["Parameters"][2]["Initial Mean"] = +1.0;


  switch( atoi(argv[1]) ){
    case 1:
      korali["Solver"]["Method"] = "CMA-ES";
      korali["Problem"]["Objective"] = "Likelihood";
      korali["Solver"]["Lambda"] = 10;
      korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-11;
      korali["Solver"]["Termination Criteria"]["Max Generations"] = 1e4;
      korali["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 3e5;
      break;

    case 2:
      korali["Solver"]["Method"] = "TMCMC";
      korali["Problem"]["Objective"] = "Posterior";
      korali["Solver"]["Covariance Scaling"] = 0.02;
      korali["Solver"]["Population Size"] = 5000;
      korali["Solver"]["Burn In"] = 5;
      korali["Solver"]["Coefficient of Variation"] = 0.5;
      break;

    default:
      printf("\nUsage: ./posterior 1(=maximize) or 2(=sample)\n");
      exit(1);
  }



  korali["Seed"] = 0xC0FFEE;
  korali["Verbosity"] = "Detailed";
  // korali["Live Plotting"] = true;


  korali.run();


 return 0;
}
