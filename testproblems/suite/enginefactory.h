#ifndef _ENGINE_FACTORY_H_
#define _ENGINE_FACTORY_H_

#include "korali.h"

namespace Suite
{


class EngineFactory
{

public:
  EngineFactory() {};
  ~EngineFactory() {};

  virtual Korali::Engine createEngine(size_t dim, double lowerbound, double upperbound, size_t maxEval, double targetFitness ) = 0;

};

class CMAESFactory : public EngineFactory
{
public:
  CMAESFactory(std::string muType, double eps) : EngineFactory(), _muType(muType), _eps(eps) {};
  ~CMAESFactory() {};
  Korali::Engine createEngine(size_t dim, double lowerbound, double upperbound, size_t maxEval, double targetFitness );

private:
  std::string _muType;
  double _eps;
};


class DEFactory : public EngineFactory
{
public:
  DEFactory(std::string parent, std::string mutationrule, std::string acceptrule) : EngineFactory (),
    _parent(parent),
    _mutationRule(mutationrule),
    _acceptRule(acceptrule) {}

  ~DEFactory() {};
  Korali::Engine createEngine(size_t dim, double lowerbound, double upperbound, size_t maxEval, double targetFitness );

private:
  std::string _parent;
  std::string _mutationRule;
  std::string _acceptRule;
};


} // namespace Suite

#endif // _ENGINE_FACTORY_H_
