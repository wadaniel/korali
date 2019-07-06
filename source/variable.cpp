#include "korali.h"
#include <chrono>

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Variable::Variable(std::string name)
{
 _name = name;
 _a = 0.0;
 _b = 0.0;
 _aux = 0.0;
 _seed = 0;
 _distributionType = KoraliDefaultDistribution;
}

Korali::Variable::~Variable()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Korali::Variable::setDistribution(nlohmann::json& js)
{
 auto dString = consume(js, { "Type" }, KORALI_STRING, "Variable");
 bool foundDistributionType = false;
 if (dString == "Cauchy")      { _distributionType = KoraliCauchyDistribution;      foundDistributionType = true; }
 if (dString == "Default")     { _distributionType = KoraliDefaultDistribution;     foundDistributionType = true; }
 if (dString == "Exponential") { _distributionType = KoraliExponentialDistribution; foundDistributionType = true; }
 if (dString == "Gamma")       { _distributionType = KoraliGammaDistribution;       foundDistributionType = true; }
 if (dString == "Gaussian")    { _distributionType = KoraliGaussianDistribution;    foundDistributionType = true; }
 if (dString == "Laplace")     { _distributionType = KoraliLaplaceDistribution;     foundDistributionType = true; }
 if (dString == "Uniform")     { _distributionType = KoraliUniformDistribution;     foundDistributionType = true; }
 if (foundDistributionType == false) { fprintf(stderr, "[Korali] Error: Incorrect or missing distribution for parameter %s.\n", _name.c_str()); exit(-1); }

 if (_distributionType == KoraliCauchyDistribution)
 {
  _a = consume(js, { "Location" }, KORALI_NUMBER);
  _b = consume(js, { "Scale" }, KORALI_NUMBER);
 }

 if (_distributionType == KoraliExponentialDistribution)
 {
  _a = consume(js, { "Mean" },     KORALI_NUMBER);
  _b = consume(js, { "Location" }, KORALI_NUMBER);
 }

 if (_distributionType == KoraliGammaDistribution)
 {
  _a = consume(js, { "Scale" }, KORALI_NUMBER);
  _b = consume(js, { "Shape" }, KORALI_NUMBER);
 }

 if (_distributionType == KoraliGaussianDistribution)
 {
  _a = consume(js, { "Mean" },  KORALI_NUMBER);
  _b = consume(js, { "Sigma" }, KORALI_NUMBER);
 }

 if (_distributionType == KoraliLaplaceDistribution)
 {
  _a = consume(js, { "Mean" },  KORALI_NUMBER);
  _b = consume(js, { "Width" }, KORALI_NUMBER);
 }

 if (_distributionType == KoraliUniformDistribution)
 {
  _a = consume(js, { "Minimum" }, KORALI_NUMBER);
  _b = consume(js, { "Maximum" }, KORALI_NUMBER);
 }

 _seed = consume(js, { "Seed" }, KORALI_NUMBER);
 _range = gsl_rng_alloc (gsl_rng_default);
 gsl_rng_set(_range, _seed);

 if (_distributionType == KoraliCauchyDistribution)      { _aux = -gsl_sf_log( _b * M_PI );}
 if (_distributionType == KoraliExponentialDistribution) { _aux = 0.0; }
 if (_distributionType == KoraliGammaDistribution)       { _aux = -gsl_sf_lngamma(_b) - _b*log(_a); }
 if (_distributionType == KoraliGaussianDistribution)    { _aux = -0.5*gsl_sf_log(2*M_PI) - gsl_sf_log(_b);}
 if (_distributionType == KoraliLaplaceDistribution)     { _aux = -gsl_sf_log(2.*_b); }
 if (_distributionType == KoraliUniformDistribution)     { _aux = -gsl_sf_log(_b-_a); }

 _isLogSpace = consume(js, { "Log Space"}, KORALI_BOOLEAN, "false");
};

void Korali::Variable::getDistribution(nlohmann::json& js)
{
 js["Seed"] = _seed;
 js["Log Space"] = _isLogSpace;

 if (_distributionType == KoraliCauchyDistribution)
 {
  js["Type"] = "Cauchy";
  js["Location"] = _a;
  js["Scale"] = _b;
 }

 if (_distributionType == KoraliExponentialDistribution)
 {
  js["Type"] = "Exponential";
  js["Mean"] = _a;
  js["Location"] = _b;
 }

 if (_distributionType == KoraliGammaDistribution)
 {
  js["Type"] = "Gamma";
  js["Scale"] = _a;
  js["Shape"] = _b;
 }

 if (_distributionType == KoraliGaussianDistribution)
 {
  js["Type"] = "Gaussian";
  js["Mean"] = _a;
  js["Sigma"] = _b;
 }

 if (_distributionType == KoraliLaplaceDistribution)
 {
  js["Type"] = "Laplace";
  js["Mean"] = _a;
  js["Width"] = _b;
 }

 if (_distributionType == KoraliUniformDistribution)
 {
  js["Type"] = "Uniform";
  js["Minimum"] = _a;
  js["Maximum"] = _b;
 }
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

double Korali::Variable::getDensity(double x)
{
 if (_distributionType == KoraliCauchyDistribution)      { return gsl_ran_cauchy_pdf( x-_a, _b ); }
 if (_distributionType == KoraliExponentialDistribution) { return gsl_ran_exponential_pdf(x-_b, _a); }
 if (_distributionType == KoraliGammaDistribution)       { return gsl_ran_gamma_pdf( x, _b, _a ); }
 if (_distributionType == KoraliGaussianDistribution)    { return gsl_ran_gaussian_pdf(x - _a, _b); }
 if (_distributionType == KoraliLaplaceDistribution)     { return gsl_ran_laplace_pdf( x-_a, _b ); }
 if (_distributionType == KoraliUniformDistribution)     { return gsl_ran_flat_pdf(x, _a, _b); }

 fprintf(stderr, "[Korali] Error: Problem requires that variable '%s' has a defined distribution.\n", _name.c_str());
 exit(-1);
 return 0.0;
};

double Korali::Variable::getLogDensity(double x)
{
 if (_distributionType == KoraliCauchyDistribution)      { return _aux - gsl_sf_log( 1. + gsl_sf_pow_int((x-_a)/_b,2) ); }
 if (_distributionType == KoraliExponentialDistribution) { if (x-_b < 0) return -INFINITY; return - log(_a) - (x-_b)/_a; }
 if (_distributionType == KoraliGammaDistribution)       { if(x < 0) return -INFINITY; return _aux + (_b-1)*log(x) - x/_a; }
 if (_distributionType == KoraliGaussianDistribution)    { double d = (x-_a)/_b; return _aux - 0.5*d*d; }
 if (_distributionType == KoraliLaplaceDistribution)     { return _aux - fabs(x-_a)/_b; }
 if (_distributionType == KoraliUniformDistribution)     { if (x >= _a && x <= _b) return _aux; return -INFINITY; }

 fprintf(stderr, "[Korali] Error: Problem requires that variable '%s' has a defined distribution.\n", _name.c_str());
 exit(-1);
 return 0.0;
};

double Korali::Variable::getRandomNumber()
{
 if (_distributionType == KoraliCauchyDistribution)      { return _a + gsl_ran_cauchy(_range, _b); }
 if (_distributionType == KoraliExponentialDistribution) { return _b + gsl_ran_exponential(_range, _a); }
 if (_distributionType == KoraliGammaDistribution)       { return gsl_ran_gamma(_range, _b, _a); }
 if (_distributionType == KoraliGaussianDistribution)    { return _a + gsl_ran_gaussian(_range, _b); }
 if (_distributionType == KoraliLaplaceDistribution)     { return _a + gsl_ran_laplace(_range, _b); }
 if (_distributionType == KoraliUniformDistribution)     { return gsl_ran_flat(_range, _a, _b); }

 fprintf(stderr, "[Korali] Error: Problem requires that variable '%s' has a defined distribution.\n", _name.c_str());
 exit(-1);
 return 0.0;
};
