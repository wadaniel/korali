#include "korali.hpp"
#include "model/model.h"
#include <stdio.h>
#include <map>

std::map<int, double> samplePHs;
std::map<int, double> sampleTemps;
std::map<int, double> bananaCounts;
std::map<int, double> orangeCounts;
std::map<int, double> appleCounts;
std::map<int, double> harvestQualities;

void storeResults(Korali::Model& d)
{
 // Gathering sample variable values
 double ph = d.getVariable(0);
 double temp = d.getVariable(1);
 int sampleId = d.getSampleId();

 // Gathering QOIs
 int b = getBananaCount(ph, temp);
 int o = getOrangeCount(ph, temp);
 int a = getAppleCount(ph, temp);

 // Calculating objective function
 int x = getHarvestQuality(b, o, a);

 // Storing Qois
 samplePHs[sampleId] = ph;
 sampleTemps[sampleId] = temp;
 bananaCounts[sampleId] = b;
 orangeCounts[sampleId] = o;
 appleCounts[sampleId] = a;
 harvestQualities[sampleId] = x;

 // Returning evaluation to Korali
 d.addResult(x);
}

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
 k["Solver"]["Target Coefficient of Variation"] = 0.2;

 // Setting Model
 k.setModel(storeResults);

 // Running Korali
 k.run();

 // Saving quantities of interest
 printf("Saving quantities of interest in myResults.txt...\n");
 FILE *configFile = fopen("myResults.txt", "w");

 for (auto const& x : harvestQualities)
 {
   int i = x.first;
   fprintf(configFile, "%f, %f, %f, %f, %f, %f\n", samplePHs[i], sampleTemps[i], orangeCounts[i], bananaCounts[i], appleCounts[i], harvestQualities[i]);
 }
}
