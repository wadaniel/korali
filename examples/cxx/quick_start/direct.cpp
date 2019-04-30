#include "korali.h"
#include <stdlib.h>


void F(std::vector<double>& x, std::vector<double>& result){

  result.push_back( -pow( x[0]-2. , 2) );

}


int main(int argc, char* argv[])
{
  if(argc<2){
    printf("\nUsage: ./posterior 1(=maximize) or 2(=sample)\n\n");
    exit(1);
  }

  auto korali = Korali::Engine([](Korali::modelData& d) {
    F(d.getParameters(), d.getResults());
  });

  korali["Problem"]["Objective"] = "Direct Evaluation";

  korali["Parameters"][0]["Name"] = "X0";
  korali["Parameters"][0]["Type"] = "Computational";
  korali["Parameters"][0]["Distribution"] = "Uniform";
  korali["Parameters"][0]["Minimum"] = -10.0;
  korali["Parameters"][0]["Maximum"] = +10.0;

  switch( atoi(argv[1]) ){
    case 1:
      korali["Solver"]["Method"] = "CMA-ES";
      korali["Solver"]["Lambda"] = 12;
      korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-11;
      korali["Solver"]["Termination Criteria"]["Min Fitness"] = 1e-12;
      korali["Solver"]["Termination Criteria"]["Max Generations"] = 1e4;
      korali["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 1e4;
      break;

    case 2:
      korali["Solver"]["Method"] = "TMCMC";
      korali["Solver"]["Covariance Scaling"] = 0.02;
      korali["Solver"]["Population Size"] = 5000;
      korali["Solver"]["Burn In"] = 5;
      korali["Solver"]["Coefficient of Variation"] = 0.5;
      break;

    default:
      printf("\nusage: ./posterior 1(=maximize) or 2(=sample)\n");
      exit(1);
  }


  korali["Seed"] = 0xC0FFEE;
  korali["Verbosity"] = "Detailed";

  korali.run();
  return 0;
}
