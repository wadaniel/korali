#include "korali.h"
#include "model/jacobi.h"

int main(int argc, char* argv[])
{
 MPI_Init(&argc, &argv);
 auto k = Korali::Engine();

 k["Problem"] = "Bayesian";
 k["Bayesian"]["Likelihood"]["Type"] = "Reference";
 k["Bayesian"]["Likelihood"]["Model"] = "Additive Gaussian";
 k["Bayesian"]["Likelihood"]["Reference Data"] = getReferenceData();

 k["Variables"][0]["Name"] = "X0";
 k["Variables"][0]["Bayesian"]["Type"] = "Computational";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = +0.0;
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

 k["Variables"][1]["Name"] = "X1";
 k["Variables"][1]["Bayesian"]["Type"] = "Computational";
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0;
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Maximum"] = +0.0;

 k["Variables"][2]["Name"] = "Y0";
 k["Variables"][2]["Bayesian"]["Type"] = "Computational";
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Minimum"] = +1.0;
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Maximum"] = +6.0;

 k["Variables"][3]["Name"] = "Y1";
 k["Variables"][3]["Bayesian"]["Type"] = "Computational";
 k["Variables"][3]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][3]["Bayesian"]["Prior Distribution"]["Minimum"] = -4.0;
 k["Variables"][3]["Bayesian"]["Prior Distribution"]["Maximum"] = +1.0;

 k["Variables"][4]["Name"] = "Z0";
 k["Variables"][4]["Bayesian"]["Type"] = "Computational";
 k["Variables"][4]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][4]["Bayesian"]["Prior Distribution"]["Minimum"] = +2.0;
 k["Variables"][4]["Bayesian"]["Prior Distribution"]["Maximum"] = +7.0;

 k["Variables"][5]["Name"] = "Z1";
 k["Variables"][5]["Bayesian"]["Type"] = "Computational";
 k["Variables"][5]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][5]["Bayesian"]["Prior Distribution"]["Minimum"] = -3.0;
 k["Variables"][5]["Bayesian"]["Prior Distribution"]["Maximum"] = +2.0;

 k["Variables"][6]["Name"] = "Sigma";
 k["Variables"][6]["Bayesian"]["Type"] = "Statistical";
 k["Variables"][6]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][6]["Bayesian"]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][6]["Bayesian"]["Prior Distribution"]["Maximum"] = 20.0;

 k["Solver"] = "TMCMC";
 k["TMCMC"]["Covariance Scaling"] = 0.02;
 k["TMCMC"]["Population Size"] = 100;

 k["Conduit"]["MPI"]["Enabled"] = true;
 k["Conduit"]["MPI"]["Ranks Per Team"] = atoi(argv[1]);

 std::vector<double> xdata = getPointData();
 k.setModel([xdata](Korali::Model& d) { jacobi(xdata, d.getVariables(), d.getResults(), d.getComm()); });

 k.run();

 return 0;
}
