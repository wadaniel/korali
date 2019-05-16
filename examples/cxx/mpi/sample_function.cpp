#include "korali.h"
#include "model/jacobi.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Detailed";

 for (int i = 0; i < 4; i++)
 {
 korali["Problem"]["Variables"][i]["Name"] = "X" + std::to_string(i);
 korali["Problem"]["Variables"][i]["Type"] = "Computational";
 korali["Problem"]["Variables"][i]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][i]["Minimum"] = -32.0;
 korali["Problem"]["Variables"][i]["Maximum"] = +32.0;
 }

 korali["Problem"]["Type"] = "Direct";

 korali["Solver"]["Method"] = "TMCMC";
 korali["Solver"]["Covariance Scaling"] = 0.02;
 korali["Solver"]["Population Size"] = 10000;
 korali["Solver"]["Min Rho Update"] = 0.0;
 
 korali.run([](Korali::Model::MPI& d) { jacobi(d.getParameters(), d.getResults(), d.getComm()); });

 return 0;
}
