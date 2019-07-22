#include "model.h"
#include "stdio.h"
#include "math.h"

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

void saveResults()
{
 // Saving quantities of interest
 printf("Saving quantities of interest in myResults.txt...\n");
 FILE *configFile = fopen("myResults.txt", "w");

 for (auto const& x : harvestQualities)
 {
   int i = x.first;
   fprintf(configFile, "%f, %f, %f, %f, %f, %f\n", samplePHs[i], sampleTemps[i], orangeCounts[i], bananaCounts[i], appleCounts[i], harvestQualities[i]);
 }
}

double getHarvestQuality(int b, int o, int a)
{
 return b*1.2 + o*0.9 + a*1.1;
}

int getBananaCount(double ph, double temp)
{
 double a = 100;
 double b = pow(ph - 7.3, 2.0) + pow((temp - 35)*0.1, 2.0);
 double c = 2;
 double t = a * exp(-b / c);
 return floor(t);
}

int getOrangeCount(double ph, double temp)
{
 double a = 50;
 double b = pow(ph - 6.6, 2.0) + pow((temp - 27)*0.1, 2.0);
 double c = 2;
 double t = a * exp(-b / c);
 return floor(t);
}

int getAppleCount(double ph, double temp)
{
 double a = 90;
 double b = pow(ph - 8.6, 2.0) + pow((temp - 20)*0.1, 2.0);
 double c = 2;
 double t = a * exp(-b / c);
 return floor(t);
}
