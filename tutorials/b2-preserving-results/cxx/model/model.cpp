#include "model.h"
#include "math.h"

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
