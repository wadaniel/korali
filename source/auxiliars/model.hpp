#ifndef __KORALI_AUXILIARS_MODEL_HPP_
#define __KORALI_AUXILIARS_MODEL_HPP_

#include <vector>
#include "auxiliars/logger.hpp"

namespace Korali
{

class Model {

 private:

 Model* _self;
 size_t _sampleId;
 std::vector<double> _sample;
 std::vector<double> _results;

 public:

 Model() { _self = this; }

 double operator[](size_t i)
 {
  if (i >= _self->_sample.size())
   Korali::logError("Requesting index %lu, but the sample contains only %lu values.\n", i, _self->_sample.size());
  return _self->_sample[i];
 }

 size_t getSampleId() { return _self->_sampleId; }
 size_t getSampleSize() { return _self->_sample.size(); }
 size_t getResultsSize() { return _self->_results.size(); }

 void setResult(const double& result) { _self->_results.clear(); _self->_results.push_back(result); }
 void setResult(const std::vector<double>& results) { _self->_results = results; }
 std::vector<double> getResults() { return _self->_results; }
 void setSample(const size_t& sampleId, const std::vector<double>& sample) { _self->_sampleId = sampleId; _self->_sample = sample; }
};

}

#endif
