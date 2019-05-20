#include "korali.h"
#include "models/rosenbrock.h"

int main(int argc, char* argv[])
{
 
 //rosenbrock
 std::function<void(Korali::modelData&)> model = [](Korali::modelData& d) { m_rosenbrock(d.getParameters(), d.getResults()); }; 
 std::function<double(double* , size_t)> fc = [](double* arr, size_t N) { return -1.0; }; 

 int    nParams = 2; 
 double lower = -32.0; 
 double upper = 32.0;

 double* tmp = (double*) malloc(0);
 auto korali = Korali::Engine(model);
 korali.addConstraint(fc);

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Detailed";
 korali["Conduit"]["Type"] = "Sequential";
 korali["Solver"]["Viability Regime"] = true;

 for (int i = 0; i < nParams; i++)
 {
 korali["Problem"]["Variables"][i]["Name"] = "X" + std::to_string(i);
 korali["Problem"]["Variables"][i]["Type"] = "Computational";
 korali["Problem"]["Variables"][i]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][i]["Minimum"] = lower;
 korali["Problem"]["Variables"][i]["Maximum"] = upper;
 }

 korali["Problem"]["Type"] = "Direct";

 korali["Solver"]["Method"] = "CCMA-ES";
 korali["Solver"]["Num Samples"] = 10;
 korali["Solver"]["Num Viability Samples"] = 4;
 korali["Solver"]["Termination Criteria"]["Max Generations"] = 100;
 korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-12;
 
 korali.run();

 std::vector<double> res = {1.0, 1.0};
 m_rosenbrock(res, res);
 printf(" rosenbrock (%f,%f) = %f\n", res.at(0), res.at(1), res.at(2) );
 return 0;

}
