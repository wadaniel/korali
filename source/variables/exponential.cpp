#include "korali.h"

using namespace Korali::Variable;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Exponential::Exponential(double mean, double loc, size_t seed)
{
 _seed = seed;
 _loc = loc;
 _mean = mean;
 initialize();
}

Exponential::Exponential(nlohmann::json& js, size_t seed)
{
 _seed = seed;
 setConfiguration(js);
 initialize();
}

Exponential::~Exponential()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Exponential::getConfiguration()
{
 auto js = nlohmann::json();

 js["Name"] = _name;
 js["Distribution"]["Distribution"] = "Exponential";
 js["Distribution"]["Location"] = _loc;
 js["Distribution"]["Mean"] = _mean;

 return js;
}

void Exponential::setConfiguration(nlohmann::json& js)
{
 _name = consume(js, { "Name" }, KORALI_STRING);
 _mean = consume(js, { "Distribution", "Mean" }, KORALI_NUMBER);
 _loc  = consume(js, { "Distribution", "Location" }, KORALI_NUMBER);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Exponential::initialize()
{
 _range = gsl_rng_alloc (gsl_rng_default);
 gsl_rng_set(_range, _seed);
}

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
