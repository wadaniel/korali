#ifndef _KORALI_CONDUIT_BASE_HPP_
#define _KORALI_CONDUIT_BASE_HPP_

#include "auxiliars/json.hpp"

namespace Korali { namespace Conduit {

class Base : public Korali::Module {

 public:

 virtual void requestEvaluation(double* target, size_t targetId) = 0;
 virtual void checkProgress() { }
 virtual bool isRoot() { return true; }
 virtual void abort() { exit(-1); }

};

class Conduit;

} } // namespace Korali::Conduit


#endif // _KORALI_CONDUIT_BASE_HPP_

