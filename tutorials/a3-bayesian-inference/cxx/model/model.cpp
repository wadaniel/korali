#include "model.h"

void model(Korali::Model& k)
{
 std::vector<double> x = getReferencePoints();
 std::vector<double> v = k.getVariables();

 for (int i = 0; i < x.size(); i++)
 {
  double th0 = v[0];
  double th1 = v[1];
  double c = th0*x[i] + th1;
  k.addResult(c);
 }
}

std::vector<double> getReferencePoints()
{
 std::vector<double> x;
 x.push_back(1.0);
 x.push_back(2.0);
 x.push_back(3.0);
 x.push_back(4.0);
 x.push_back(5.0);
 return x;
}

std::vector<double> getReferenceData()
{
 std::vector<double> y;
 y.push_back(3.2069);
 y.push_back(4.1454);
 y.push_back(4.9393);
 y.push_back(6.0588);
 y.push_back(6.8425);
 return y;
}
