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
    printf("\nUsage: ./posterior 1(=maximize) or 2(=sample)\n\n");
    exit(1);
  }

  std::vector<double> x, y;
  x.push_back(1.00);   y.push_back(-0.2045);
  x.push_back(3.25);   y.push_back( 4.0171);
  x.push_back(5.50);   y.push_back( 9.6384);
  x.push_back(7.75);   y.push_back(14.1257);
  x.push_back(10.0);   y.push_back(16.2702);

  auto Fx = [x]( Korali::modelData& d ) {
                F(d.getParameters(), d.getResults(), x);
  };

  auto korali = Korali::Engine( Fx );


  korali["Problem"]["Objective"] = "Posterior";

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
      korali["Solver"]["Lambda"] = 10;
      korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-11;
      korali["Solver"]["Termination Criteria"]["Min Fitness"] = 1e-12;
      korali["Solver"]["Termination Criteria"]["Max Generations"] = 1e4;
      korali["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 3e5;
      break;

    case 2:
      korali["Solver"]["Method"] = "TMCMC";
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
  korali["Live Plotting"] = true;


  korali.run();


 return 0;
}
