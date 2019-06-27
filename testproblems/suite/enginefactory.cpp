#include "enginefactory.h"

namespace Suite
{

Korali::Engine CMAESFactory::createEngine(size_t dim, double lowerbound, double upperbound, size_t maxEval, double targetFitness )
{
  auto k = Korali::Engine();
  k["Verbosity"] = "Minimal";
  k["CMA-ES"]["Result Output Frequency"] = 1;

  k["Problem"] = "Direct Evaluation";
  k["Solver"]  = "CMA-ES";

  k["CMA-ES"]["Objective"]          = "Minimize";
  k["CMA-ES"]["Sample Count"]       = 4+3*log(dim);
  k["CMA-ES"]["Mu"]["Type"]         = _muType;
  k["CMA-ES"]["Mu"]["Exploitation"] = _eps;

  for (int i = 0; i < dim; i++)
  {
    k["Variables"][i]["Name"] = "X" + std::to_string(i);
    k["Variables"][i]["CMA-ES"]["Lower Bound"] = lowerbound;
    k["Variables"][i]["CMA-ES"]["Upper Bound"] = upperbound;
  }

  k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Active"]          = false;
  k["CMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"]["Active"]   = false;
  k["CMA-ES"]["Termination Criteria"]["Min Standard Deviation"]["Active"]   = false;
  k["CMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Active"]   = false;
  k["CMA-ES"]["Termination Criteria"]["Max Condition Covariance"]["Active"] = false;
  //k["CMA-ES"]["Termination Criteria"]["Min Step Size"]["Active"]            = false;

  k["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Active"] = true;
  k["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Value"]  = maxEval;

  k["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Active"] = true;
  k["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"]  = targetFitness;
  return k;
}

Korali::Engine DEFactory::createEngine(size_t dim, double lowerbound, double upperbound, size_t maxEval, double targetFitness )
{
  auto k = Korali::Engine();
  k["Verbosity"] = "Minimal";
  k["DE"]["Result Output Frequency"] = 5000;

  k["Problem"] = "Direct Evaluation";
  k["Solver"]  = "DE";

  k["DE"]["Objective"]    = "Minimize";
  k["DE"]["Sample Count"] = 10*dim;

  k["DE"]["Parent"] = _parent;
  k["DE"]["Mutation Rule"] = _mutationRule;
  k["DE"]["Accept Rule"] = _acceptRule;

  for (int i = 0; i < dim; i++)
  {
    k["Variables"][i]["Name"] = "X" + std::to_string(i);
    k["Variables"][i]["DE"]["Lower Bound"] = lowerbound;
    k["Variables"][i]["DE"]["Upper Bound"] = upperbound;
  }

  k["DE"]["Termination Criteria"]["Max Generations"]["Active"]        = false;
  k["DE"]["Termination Criteria"]["Fitness Diff Threshold"]["Active"] = false;
  k["DE"]["Termination Criteria"]["Min DeltaX"]["Active"]             = false;

  k["DE"]["Termination Criteria"]["Max Model Evaluations"]["Active"] = true;
  k["DE"]["Termination Criteria"]["Max Model Evaluations"]["Value"]  = maxEval;

  k["DE"]["Termination Criteria"]["Fitness"]["Active"] = true;
  k["DE"]["Termination Criteria"]["Fitness"]["Value"]  = targetFitness;

  return k;
}

} // namepsace Suite
