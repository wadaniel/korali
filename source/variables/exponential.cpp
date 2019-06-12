#include "korali.h"

using namespace Korali::Variable;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Exponential::Exponential(double mean, double loc, size_t seed) : Base::Base(seed)
{
	_loc = loc;
	_mean = mean;
}

Exponential::Exponential(nlohmann::json& js, size_t seed) : Base::Base(js, seed)
{
 setConfiguration(js);
}

Exponential::~Exponential()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Exponential::getConfiguration()
{
 auto js = this->Base::getConfiguration();

 js["Distribution"]["Distribution"] = "Exponential";
 js["Distribution"]["Location"] = _loc;
 js["Distribution"]["Mean"] = _mean;

 return js;
}

void Exponential::setConfiguration(nlohmann::json& js)
{
 _mean = consume(js, { "Distribution", "Mean" }, KORALI_NUMBER);
 _loc  = consume(js, { "Distribution", "Location" }, KORALI_NUMBER);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

double Exponential::getDensity(double x)
{
 return gsl_ran_exponential_pdf(x-_loc, _mean);
}

double Exponential::getLogDensity(double x)
{
 if (x-_loc < 0) return -INFINITY;
 return - log(_mean) - (x-_loc)/_mean;
}

double Exponential::getRandomNumber()
{
 return _loc + gsl_ran_exponential(_range, _mean);
}

void Exponential::printDetails()
{
  printf("Exponential(%.3g,%.3g)", _loc, _mean);
}
