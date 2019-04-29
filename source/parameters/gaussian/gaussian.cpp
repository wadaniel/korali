#include "korali.h"

using namespace Korali::Parameter;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Gaussian::Gaussian(double mean, double sigma, size_t seed) : Base::Base(seed)
{
	_mean = mean;
	_sigma = sigma;
	initialize();
}

Gaussian::Gaussian(nlohmann::json& js, size_t seed) : Base::Base(js, seed)
{
 setConfiguration(js);
 initialize();
}

Gaussian::~Gaussian()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Gaussian::getConfiguration()
{
 auto js = this->Base::getConfiguration();

 js["Distribution"] = "Gaussian";
 js["Mean"] = _mean;
 js["Sigma"] = _sigma;

 return js;
}

void Gaussian::setConfiguration(nlohmann::json& js)
{
 _mean  = consume(js, { "Mean" }, KORALI_NUMBER);
 _sigma = consume(js, { "Sigma" }, KORALI_NUMBER);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Gaussian::initialize()
{
	_aux = -0.5*gsl_sf_log(2*M_PI) - gsl_sf_log(_sigma);
}

double Gaussian::getDensity(double x)
{
 return gsl_ran_gaussian_pdf(x - _mean, _sigma);
}

double Gaussian::getDensityLog(double x)
{
 double d = (x-_mean)/_sigma;
 return _aux - 0.5*d*d;
}

double Gaussian::getRandomNumber()
{
 return _mean + gsl_ran_gaussian(_range, _sigma);
}

double Gaussian::logLikelihood(double sigma, int nData, double* x, double* u)
{
 double sigma2 = sigma*sigma;
 double ssn = 0.0;

 for(int i = 0; i < nData; i++)
  {
  double diff = x[i] - u[i];
  ssn += diff*diff;
  }

 double res = 0.5* (nData*log(2*M_PI) + nData*log(sigma2) + ssn/sigma2);

 return res;
}

void Gaussian::printDetails()
{
  printf("Gaussian(%.3g,%.3g)", _mean, _sigma);
}
