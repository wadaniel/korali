#include <korali.hpp>
#include <map>

extern std::map<int, double> samplePHs;
extern std::map<int, double> sampleTemps;
extern std::map<int, double> bananaCounts;
extern std::map<int, double> orangeCounts;
extern std::map<int, double> appleCounts;
extern std::map<int, double> harvestQualities;

void storeResults(Korali::Model& d);
void saveResults();

double getHarvestQuality(int b, int o, int a);
int getBananaCount(double ph, double temp);
int getOrangeCount(double ph, double temp);
int getAppleCount(double ph, double temp);
