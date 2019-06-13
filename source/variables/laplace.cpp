#include "korali.h"

using namespace Korali::Variable;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Laplace::Laplace(double mean, double width, size_t seed) : Base::Base(seed)
{
 _mean  = mean;
 _width = width;
 initialize();
}

Laplace::Laplace(nlohmann::json& js, size_t seed) : Base::Base(js, seed)
{
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
 auto js = this->Base::getConfiguration();

 js["Distribution"]["Type"]  = "Laplace";
 js["Distribution"]["Mean"]  = _mean;
 js["Distribution"]["Width"] = _width;

 return js;
}

void Laplace::setConfiguration(nlohmann::json& js)
{
 _mean  = consume(js, { "Distribution", "Mean" }, KORALI_NUMBER);
 _width = consume(js, { "Distribution", "Width" }, KORALI_NUMBER);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Laplace::initialize()
{
 _aux = - gsl_sf_log(2.*_width);
 _hasDistribution = true;
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
