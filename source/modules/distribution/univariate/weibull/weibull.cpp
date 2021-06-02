#include "modules/distribution/univariate/weibull/weibull.hpp"
#include "modules/experiment/experiment.hpp"
#include <gsl/gsl_randist.h>

namespace korali
{
namespace distribution
{
namespace univariate
{


double Weibull::getDensity(const double x) const
{
  return gsl_ran_weibull_pdf(x, _scale, _shape);
}

double Weibull::getLogDensity(const double x) const
{
  
  if (x <= 0.0) return -INFINITY;
  
  // To-do: Fix this too
  // return _aux + (_scale - 1.) * std::log(x) - std::pow((x / _scale), _shape);
  return std::log(getDensity(x));
}

double Weibull::getLogDensityGradient(const double x) const
{
  if (x <= 0.0) return 0.0;
  return ((_scale - 1.) - _shape * std::pow((x / _scale), _shape)) / x;
}

double Weibull::getLogDensityHessian(const double x) const
{
  if (x <= 0.0) return 0.0;
  return ((1. - _scale) + _shape * std::pow((x / _scale), _shape) - _shape * _shape * std::pow((x / _scale), _shape)) / (x * x);
}

double Weibull::getRandomNumber()
{
  return gsl_ran_weibull(_range, _scale, _shape);
}

void Weibull::updateDistribution()
{
  if (_shape <= 0.0) KORALI_LOG_ERROR("Incorrect Shape parameter of Weibull distribution: %f.\n", _shape);
  if (_scale <= 0.0) KORALI_LOG_ERROR("Incorrect Scale parameter of Weibull distribution: %f.\n", _scale);

  _aux = log(_shape / _scale) - (_shape - 1.0) * log(_scale);
}

void Weibull::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Shape"].is_number()) _shape = js["Shape"];
 if(js["Shape"].is_string()) { _hasConditionalVariables = true; _shapeConditional = js["Shape"]; } 
 eraseValue(js, "Shape");

 if(js["Scale"].is_number()) _scale = js["Scale"];
 if(js["Scale"].is_string()) { _hasConditionalVariables = true; _scaleConditional = js["Scale"]; } 
 eraseValue(js, "Scale");

 Univariate::setConfiguration(js);
 _type = "univariate/weibull";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: weibull: \n%s\n", js.dump(2).c_str());
} 

void Weibull::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_shapeConditional == "") js["Shape"] = _shape;
 if(_shapeConditional != "") js["Shape"] = _shapeConditional; 
 if(_scaleConditional == "") js["Scale"] = _scale;
 if(_scaleConditional != "") js["Scale"] = _scaleConditional; 
 Univariate::getConfiguration(js);
} 

void Weibull::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void Weibull::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* Weibull::getPropertyPointer(const std::string& property)
{
 if (property == "Shape") return &_shape;
 if (property == "Scale") return &_scale;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution Weibull.\n", property.c_str());
 return NULL;
}



} //univariate
} //distribution
} //korali

