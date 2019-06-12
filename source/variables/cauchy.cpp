#include "korali.h"

using namespace Korali::Variable;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Cauchy::Cauchy(double scale, double loc, size_t seed) : Base::Base(seed)
{
	_loc = loc;
	_scale = scale;
	initialize();
}

Cauchy::Cauchy(nlohmann::json& js, size_t seed) : Base::Base(js, seed)
{
 setConfiguration(js);
 initialize();
}

Cauchy::~Cauchy()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Cauchy::getConfiguration()
{
 auto js = this->Base::getConfiguration();

 js["Distribution"]["Type"] = "Cauchy";
 js["Distribution"]["Location"] = _loc;
 js["Distribution"]["Scale"] = _scale;

 return js;
}

void Cauchy::setConfiguration(nlohmann::json& js)
{
 _loc  = consume(js, { "Distribution", "Location" }, KORALI_NUMBER);
 _scale = consume(js, { "Distribution", "Scale" }, KORALI_NUMBER);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Cauchy::initialize()
{
 _aux = -gsl_sf_log( _scale * M_PI );
}

double Cauchy::getDensity(double x)
{
 return gsl_ran_cauchy_pdf( x-_loc, _scale );
}

double Cauchy::getLogDensity(double x)
{
 return  _aux - gsl_sf_log( 1. + gsl_sf_pow_int((x-_loc)/_scale,2) );
}

double Cauchy::getRandomNumber()
{
 return _loc + gsl_ran_cauchy(_range, _scale);
}

void Cauchy::printDetails()
{
  printf("Cauchy(%.3g,%.3g)", _loc, _scale);
}
