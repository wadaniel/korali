#include "posteriorModel.h"

void posteriorModel(std::vector<double>& pars, std::vector<double>& result, std::vector<double> x)
{
 for (int i = 0; i < x.size(); i++)
 {
  double th0 = pars[0];
  double th1 = pars[1];
  double r = th0*x[i] + th1;
  result.push_back(r);
 }
}

void getReferenceData(std::vector<double>& x, std::vector<double>& y)
{
 x.push_back(1.0); y.push_back(3.2069);
 x.push_back(2.0); y.push_back(4.1454);
 x.push_back(3.0); y.push_back(4.9393);
 x.push_back(4.0); y.push_back(6.0588);
 x.push_back(5.0); y.push_back(6.8425);
}

