#ifndef _KORALI_DISTRIBUTION_BASE_HPP_
#define _KORALI_DISTRIBUTION_BASE_HPP_

#include "json.hpp"
#include <vector>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>

namespace Korali { namespace Distribution {

class Base {

 private:

 size_t _seed;
 gsl_rng* _range;
 double _aux;

 public:

 virtual void initialize();
 virtual void finalize();
 static Base* getDistribution(nlohmann::json& js);
 virtual setProperty(std::string propertyName, double value) = 0;
 virtual updateDistribution(std::vector<double> sample) = 0;
 virtual double getDensity(double x) = 0;
 virtual double getLogDensity(double x) = 0;
 virtual double getRandomNumber() = 0;
 virtual void setConfiguration(nlohmann::json& js) = 0;
 virtual void getConfiguration(nlohmann::json& js) = 0;
};

} } // namespace Korali::Distribution

#endif // _KORALI_DISTRIBUTION_BASE_HPP_
