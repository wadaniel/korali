#include "fMadGrad.hpp"
#include <cmath>
#include <cstdlib>
#include <stdio.h>

namespace korali
{

fMadGrad::fMadGrad(size_t nVars) : fGradientBasedOptimizer(nVars)
{
  // Defaults
  _eta = 0.001f;
  _epsilon = 1e-08f;

  _s.resize(nVars);
  _v.resize(nVars);
  _z.resize(nVars);
  reset();
}

void fMadGrad::reset()
{
  _currentGeneration = 1;
  _modelEvaluationCount = 0;

  for (size_t i = 0; i < _nVars; i++)
    if (std::isfinite(_initialValues[i]) == false)
    {
      fprintf(stderr, "Initial Value of variable \'%lu\' not defined (no defaults can be calculated).\n", i);
      std::abort();
    }

  for (size_t i = 0; i < _nVars; i++)
  {
    _currentValue[i] = _initialValues[i];
    _s[i] = 0.0f;
    _v[i] = 0.0f;
    _z[i] = 0.0f;
  }

  _bestEvaluation = +std::numeric_limits<float>::infinity();
}

void fMadGrad::processResult(float evaluation, std::vector<float> &gradient)
{
  if (gradient.size() != _nVars)
  {
    fprintf(stderr, "Size of sample's gradient evaluations vector (%lu) is different from the number of problem variables defined (%lu).\n", gradient.size(), _nVars);
    std::abort();
  }

  float lambda = _eta * std::sqrt((float)_modelEvaluationCount + 1.0f);
  float momentum = 1.0f / ( (float)_modelEvaluationCount + 1.0f); // There can be different momentum types

  for (size_t i = 0 ; i < _nVars; i++)
  {
    _s[i] = _s[i] - lambda * gradient[i];
    _v[i] = _v[i] + lambda * (gradient[i] * gradient[i]);
    _z[i] = _initialValues[i] - (1.0f / (std::cbrt(_v[i]) + _epsilon)) * _s[i];
    _currentValue[i] = (1.0f - momentum) * _currentValue[i] +  momentum * _z[i];
  }

  _modelEvaluationCount++;
}

bool fMadGrad::checkTermination()
{
  if (_currentGeneration >= _maxGenerations) return true;

  return false;
}

void fMadGrad::printInfo()
{
  printf("x = [ ");
  for (size_t k = 0; k < _nVars; k++) printf(" %.5le  ", _currentValue[k]);
  printf(" ]\n");

  printf("F(X) = %le \n", _currentEvaluation);

  printf("DF(X) = [ ");
  for (size_t k = 0; k < _nVars; k++) printf(" %.5le  ", _gradient[k]);
  printf(" ]\n");
}

} // namespace korali
