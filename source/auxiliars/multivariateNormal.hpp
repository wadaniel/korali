#ifndef _KORALI_AUXILIARS_MULTIVARIATE_NORMAL_HPP_
#define _KORALI_AUXILIARS_MULTIVARIATE_NORMAL_HPP_

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <auxiliars/json.hpp>

namespace Korali {

class MultivariateNormal {

 private:
 std::vector<double> _meanVector;
 std::vector<double> _covarianceMatrix;
 std::vector<double> _workVector;

 gsl_matrix_view _sigma_view;
 gsl_vector_view _mean_view;
 gsl_vector_view _work_view;

 gsl_rng* _range;

 public:

 MultivariateNormal(size_t seed);
 ~MultivariateNormal();

 void updateDistribution();
 void setProperty(std::string propertyName, std::vector<double> values);
 void getDensityVector(double *x, double* result, size_t n);
 void getLogDensityVector(double *x, double* result, size_t n);
 void getRandomVector(double *x, size_t n);
};

} // namespace Korali

#endif // _KORALI_AUXILIARS_MULTIVARIATE_NORMAL_HPP_
