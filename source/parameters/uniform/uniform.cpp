#include "korali.h"

using namespace Korali::Parameter;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Uniform::Uniform(double lowerBound, double upperBound, size_t seed) : Base::Base(seed)
{
   _lowerBound = lowerBound;
   _upperBound = upperBound;
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

 js["Distribution"] = "Uniform";

 return js;
}

void Uniform::setConfiguration(nlohmann::json& js)
{
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Uniform::initialize()
{
	_aux = -gsl_sf_log(_upperBound-_lowerBound);
}

double Uniform::getDensity(double x)
{
 return gsl_ran_flat_pdf(x, _lowerBound, _upperBound);
}

double Uniform::getDensityLog(double x)
{
 if (x >= _lowerBound && x <= _upperBound)
  return _aux;
 return -GSL_NEGINF;
}

double Uniform::getRandomNumber()
{
 return gsl_ran_flat(_range, _lowerBound, _upperBound);
}

void Uniform::printDetails()
{
  printf("Uniform(%.3g,%.3g)", _lowerBound, _upperBound);
}
