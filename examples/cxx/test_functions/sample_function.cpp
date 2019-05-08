#include "korali.h"
#include "models/ackley.h"
#include "models/eggholder.h"
#include "models/gaussian.h"
#include "models/levi13.h"
#include "models/rosenbrock.h"
#include "models/shekel.h"

int main(int argc, char* argv[])
{
  
  if( argc<2 ){
    printf("\nInvalid Args! Usage:\n"
           "\t./sample_function 1 (=sample exp(-ackley))\n"
           "\t./sample_function 2 (=sample exp(-eggholder))\n"
           "\t./sample_function 3 (=sample gaussian))\n"
           "\t./sample_function 4 (=sample exp(-levi13))\n"
           "\t./sample_function 5 (=sample exp(-rosenbrock))\n"
           "\t./sample_function 6 (=sample exp(-shekel)\n\n");
    exit(1);
  }

 std::function<void(Korali::modelData&)> model;

 int nParams;
 double lower, upper;
 switch( atoi(argv[1]) ){
 case 1:
  nParams = 4; lower = -32.0; upper = 32.0;
  model = [](Korali::modelData& d) { m_ackley(d.getParameters(), d.getResults()); };
  break;

 case 2:
  nParams = 2; lower = -512.0; upper = 512.0;
  model = [](Korali::modelData& d) { m_eggholder(d.getParameters(), d.getResults()); };
  break;

    case 3:
        nParams = 5; lower = -10.0; upper = 10.0;
        gaussian_init(nParams);
        model = [](Korali::modelData& d) { gaussian(d.getParameters(), d.getResults()); }; 
        break;

    case 4:
        nParams = 2; lower = -10.0; upper = 10.0;
        model = [](Korali::modelData& d) { m_levi13(d.getParameters(), d.getResults()); }; 
        break;

    case 5: 
        nParams = 2; lower = -32.0; upper = 32.0;
        model = [](Korali::modelData& d) { m_rosenbrock(d.getParameters(), d.getResults()); }; 
        break;

    case 6:  
        nParams = 4; lower = 0.0; upper = 10.0;
        model = [](Korali::modelData& d) { m_shekel(d.getParameters(), d.getResults()); }; 
        break;

    default:
        printf("\nInvalid Args! Usage:\n"
           "\t./sample_function 1 (=sample exp(-ackley))\n"
           "\t./sample_function 2 (=sample exp(-eggholder))\n"
           "\t./sample_function 3 (=sample gaussian)\n"
           "\t./sample_function 4 (=sample exp(-levi13))\n"
           "\t./sample_function 5 (=sample exp(-rosenbrock))\n"
           "\t./sample_function 6 (=sample exp(-shekel)\n\n");
        exit(1);
 
 }

 auto korali = Korali::Engine( model );

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Detailed";

 for (int i = 0; i < nParams; i++)
 {
 korali["Parameters"][i]["Name"] = "X" + std::to_string(i);
 korali["Parameters"][i]["Type"] = "Computational";
 korali["Parameters"][i]["Distribution"] = "Uniform";
 korali["Parameters"][i]["Minimum"] = lower;
 korali["Parameters"][i]["Maximum"] = upper;
 }

 korali["Problem"]["Type"] = "Direct";

 korali["Solver"]["Method"] = "TMCMC";
 korali["Solver"]["Covariance Scaling"] = 0.02;
 korali["Solver"]["Population Size"] = 10000;
 
 korali.run();

 return 0;
}
