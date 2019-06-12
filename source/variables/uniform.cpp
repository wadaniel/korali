#include "korali.h"

using namespace Korali::Variable;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Uniform::Uniform(double minimum, double maximum, size_t seed) : Base::Base(seed)
{
   _minimum = minimum;
   _maximum = maximum;
   initialize();
}

Uniform::Uniform(nlohmann::json& js, size_t seed) : Base::Base(js, seed)
{
 setConfiguration(js);
 initialize();
}

Uniform::~Uniform()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Uniform::getConfiguration()
{
 auto js = this->Base::getConfiguration();

 js["Distribution"]["Type"] = "Uniform";
 js["Distribution"]["Minimum"] = _minimum;
 js["Distribution"]["Maximum"] = _maximum;

 return js;
}

void Uniform::setConfiguration(nlohmann::json& js)
{
 _minimum  = consume(js, { "Distribution", "Minimum" }, KORALI_NUMBER);
 _maximum = consume(js, { "Distribution", "Maximum" }, KORALI_NUMBER);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Uniform::initialize()
{
 _aux = -gsl_sf_log(_maximum-_minimum);
}

double Uniform::getDensity(double x)
{
 return gsl_ran_flat_pdf(x, _minimum, _maximum);
}

double Uniform::getLogDensity(double x)
{
 if (x >= _minimum && x <= _maximum)
  return _aux;
 return -GSL_NEGINF;
}

double Uniform::getRandomNumber()
{
 return gsl_ran_flat(_range, _minimum, _maximum);
}

void Uniform::printDetails()
{
  printf("Uniform(%.3g,%.3g)", _minimum, _maximum);
}
