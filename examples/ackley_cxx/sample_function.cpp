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

 k["Problem"]["Variables"][0]["Distribution"] = "Uniform";
 k["Problem"]["Variables"][1]["Distribution"] = "Uniform";
 k["Problem"]["Variables"][2]["Distribution"] = "Uniform";
 k["Problem"]["Variables"][3]["Distribution"] = "Uniform";

 k["Problem"]["Variables"][0]["Minimum"] = -32.0;
 k["Problem"]["Variables"][1]["Minimum"] = -32.0;
 k["Problem"]["Variables"][2]["Minimum"] = -32.0;
 k["Problem"]["Variables"][3]["Minimum"] = -32.0;

 k["Problem"]["Variables"][0]["Maximum"] = +32.0;
 k["Problem"]["Variables"][1]["Maximum"] = +32.0;
 k["Problem"]["Variables"][2]["Maximum"] = +32.0;
 k["Problem"]["Variables"][3]["Maximum"] = +32.0;

 k["Solver"]["Method"] = "TMCMC";
 k["Solver"]["Covariance Scaling"] = 0.02;
 k["Solver"]["Population Size"] = 10000;
 k["Solver"]["Min Rho Update"] = 0.0;

 k.setModel([](Korali::ModelData& d) { m_ackley(d.getVariables(), d.getResults()); });

 k.run();

 return 0;
}
