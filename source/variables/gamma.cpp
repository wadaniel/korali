#include "korali.h"

using namespace Korali::Variable;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Gamma::Gamma(double shape, double scale, size_t seed)
{
 _seed = seed;
 _shape = shape;
 _scale = scale;
 initialize();
}

Gamma::Gamma(nlohmann::json& js, size_t seed)
{
 _seed = seed;
 setConfiguration(js);
 initialize();
}

Gamma::~Gamma()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Gamma::getConfiguration()
{
 auto js = nlohmann::json();

 js["Name"] = _name;
 js["Distribution"]["Type"] = "Gamma";
 js["Distribution"]["Scale"] = _scale;
 js["Distribution"]["Shape"] = _shape;

 return js;
}

void Gamma::setConfiguration(nlohmann::json& js)
{
 _name = consume(js, { "Name" }, KORALI_STRING);
 _scale  = consume(js, { "Distribution", "Scale" }, KORALI_NUMBER);
 _shape = consume(js, { "Distribution", "Shape" }, KORALI_NUMBER);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Gamma::initialize()
{
 _range = gsl_rng_alloc (gsl_rng_default);
 gsl_rng_set(_range, _seed);
 _aux = - gsl_sf_lngamma(_shape) - _shape*log(_scale);
}

double Gamma::getDensity(double x)
{
 return gsl_ran_gamma_pdf( x, _shape, _scale );
}

double Gamma::getLogDensity(double x)
{
 if(x < 0) return -INFINITY;
 return _aux + (_shape-1)*log(x) - x/_scale;
}

double Gamma::getRandomNumber()
{
 return gsl_ran_gamma(_range, _shape, _scale);
}

void Gamma::printDetails()
{
  printf("Gamma(%.3g,%.3g)", _shape, _scale);
}
