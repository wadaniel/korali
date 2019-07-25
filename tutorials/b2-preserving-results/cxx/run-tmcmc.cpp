#include "korali.hpp"
#include "model/model.h"
#include <stdio.h>

int main(int argc, char* argv[])
{
 // Starting Korali's Engine
 auto k = Korali::Engine();

 // Selecting problem and solver types.
 k["Problem"]["Type"] = "Sampling";

 // Defining the problem's variables
 k["Variables"][0]["Name"] = "Soil pH";
 k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Prior Distribution"]["Minimum"] = 3.0;
 k["Variables"][0]["Prior Distribution"]["Maximum"] = 9.0;

 k["Variables"][1]["Name"] = "Average Temperature (C)";
 k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Prior Distribution"]["Minimum"] = +10.0;
 k["Variables"][1]["Prior Distribution"]["Maximum"] = +50.0;

 // Configuring the TMCMC sampler parameters
 k["Solver"]["Type"] = "TMCMC";
 k["Solver"]["Population Size"] = 5000;

 // Setting Model
 k.setModel(storeResults);

 // Running Korali
 k.run();

 // Saving Results
 saveResults();
}
