#include "korali.h"
#include "model/ackley.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"]["Evaluation Type"] = "Direct";

 k["Problem"]["Variables"][0]["Name"] = "X0";
 k["Problem"]["Variables"][1]["Name"] = "X1";
 k["Problem"]["Variables"][2]["Name"] = "X2";
 k["Problem"]["Variables"][3]["Name"] = "X3";

 k["Problem"]["Variables"][0]["Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][1]["Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][2]["Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][3]["Distribution"]["Type"] = "Uniform";

 k["Problem"]["Variables"][0]["Distribution"]["Minimum"] = -32.0;
 k["Problem"]["Variables"][1]["Distribution"]["Minimum"] = -32.0;
 k["Problem"]["Variables"][2]["Distribution"]["Minimum"] = -32.0;
 k["Problem"]["Variables"][3]["Distribution"]["Minimum"] = -32.0;

 k["Problem"]["Variables"][0]["Distribution"]["Maximum"] = +32.0;
 k["Problem"]["Variables"][1]["Distribution"]["Maximum"] = +32.0;
 k["Problem"]["Variables"][2]["Distribution"]["Maximum"] = +32.0;
 k["Problem"]["Variables"][3]["Distribution"]["Maximum"] = +32.0;

 k["Solver"]["Method"] = "MCMC";
 k["Solver"]["Covariance Scaling"] = 0.02;
 k["Solver"]["Population Size"] = 10000;
 k["Solver"]["Min Rho Update"] = 0.0;

 k.setModel([](Korali::ModelData& d) { m_ackley(d.getVariables(), d.getResults()); });

 k.run();

 return 0;
}
