#include "korali.h"

using namespace Korali::Variable;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Laplace::Laplace(double mean, double width, size_t seed)
{
 _seed = seed;
 _mean  = mean;
 _width = width;
 initialize();
}

Laplace::Laplace(nlohmann::json& js, size_t seed)
{
 _seed = seed;
 setConfiguration(js);
 initialize();
}

Laplace::~Laplace()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Laplace::getConfiguration()
{
 auto js = nlohmann::json();

 js["Name"] = _name;
 js["Distribution"]["Type"]  = "Laplace";
 js["Distribution"]["Mean"]  = _mean;
 js["Distribution"]["Width"] = _width;

 return js;
}

void Laplace::setConfiguration(nlohmann::json& js)
{
 _name = consume(js, { "Name" }, KORALI_STRING);
 _mean  = consume(js, { "Distribution", "Mean" }, KORALI_NUMBER);
 _width = consume(js, { "Distribution", "Width" }, KORALI_NUMBER);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Laplace::initialize()
{
 _range = gsl_rng_alloc (gsl_rng_default);
 gsl_rng_set(_range, _seed);
 _aux = - gsl_sf_log(2.*_width);
}

double Laplace::getDensity(double x)
{
 return gsl_ran_laplace_pdf( x-_mean, _width );
}

double Laplace::getLogDensity(double x)
{
 return  _aux - fabs(x-_mean)/_width;
}

double Laplace::getRandomNumber()
{
 return _mean + gsl_ran_laplace(_range, _width);
}

void Laplace::printDetails()
{
  printf("Laplace(%.3g,%.3g)", _mean, _width);
}
