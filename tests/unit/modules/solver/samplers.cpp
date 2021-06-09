#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/sampler/sampler.hpp"
#include "modules/problem/sampling/sampling.hpp"
#include "modules/problem/bayesian/reference/reference.hpp"
#include "modules/solver/sampler/MCMC/MCMC.hpp"
#include "modules/solver/sampler/TMCMC/TMCMC.hpp"

namespace
{
 using namespace korali;
 using namespace korali::solver;
 using namespace korali::solver::sampler;
 using namespace korali::problem;
 using namespace korali::problem::bayesian;

 //////////////// BASE CLASS ////////////////////////

  TEST(samplers, baseClass)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();

   // Creating initial variable
   Variable v;
   e._variables.push_back(&v);
   e["Variables"][0]["Name"] = "Var 1";
   e["Variables"][0]["Initial Mean"] = 0.0;
   e["Variables"][0]["Initial Standard Deviation"] = 0.25;
   e["Variables"][0]["Lower Bound"] = -1.0;
   e["Variables"][0]["Upper Bound"] = 1.0;

   // Creating optimizer configuration Json
   knlohmann::json samplerJs;
   samplerJs["Type"] = "Sampler/MCMC";

   // Creating module
   MCMC* opt;
   ASSERT_NO_THROW(opt = dynamic_cast<MCMC *>(Module::getModule(samplerJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(opt->applyModuleDefaults(samplerJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(opt->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = samplerJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   // Testing optional parameters
   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Lower Bound");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Lower Bound"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Lower Bound"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Upper Bound");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Upper Bound"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Upper Bound"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Initial Value");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Value"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Value"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));
  }

  //////////////// MCMC CLASS ////////////////////////

  TEST(samplers, MCMC)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();

   // Creating initial variable
   Variable v;
   e._variables.push_back(&v);
   e["Variables"][0]["Name"] = "Var 1";
   e["Variables"][0]["Initial Mean"] = 0.0;
   e["Variables"][0]["Initial Standard Deviation"] = 0.25;
   e["Variables"][0]["Lower Bound"] = -1.0;
   e["Variables"][0]["Upper Bound"] = 1.0;

   // Creating optimizer configuration Json
   knlohmann::json samplerJs;
   samplerJs["Type"] = "Sampler/MCMC";

   // Creating module
   MCMC* opt;
   ASSERT_NO_THROW(opt = dynamic_cast<MCMC *>(Module::getModule(samplerJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(opt->applyModuleDefaults(samplerJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(opt->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = samplerJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   // Testing optional parameters
   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Cholesky Decomposition Covariance"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Cholesky Decomposition Covariance"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Cholesky Decomposition Chain Covariance"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Cholesky Decomposition Chain Covariance"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leader Evaluation"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leader Evaluation"] = 0.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Evaluations"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Evaluations"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Rejection Alphas"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Rejection Alphas"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Rate"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Rate"] = 0.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Count"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Count"] = 0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Proposed Sample Count"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Proposed Sample Count"] = 0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Database"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Database"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Mean"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Mean"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Covariance Placeholder"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Covariance Placeholder"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Covariance"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Covariance"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Length"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Length"] = 0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Burn In");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Burn In"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Burn In"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Leap");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Leap"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Leap"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Rejection Levels");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Rejection Levels"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Rejection Levels"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Use Adaptive Sampling");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Use Adaptive Sampling"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Use Adaptive Sampling"] = true;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Non Adaption Period");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Non Adaption Period"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Non Adaption Period"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Chain Covariance Scaling");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Covariance Scaling"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Covariance Scaling"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"].erase("Max Samples");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Samples"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Samples"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   ///// Variable Tests

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Initial Mean");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Mean"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Mean"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Initial Standard Deviation");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Standard Deviation"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Standard Deviation"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));
  }

  //////////////// TMCMC CLASS ////////////////////////

  TEST(samplers, TMCMC)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();

   // Creating initial variable
   Variable v;
   e._variables.push_back(&v);
   e["Variables"][0]["Name"] = "Var 1";
   e["Variables"][0]["Initial Mean"] = 0.0;
   e["Variables"][0]["Initial Standard Deviation"] = 0.25;
   e["Variables"][0]["Lower Bound"] = -1.0;
   e["Variables"][0]["Upper Bound"] = 1.0;

   // Configuring Problem
   e["Problem"]["Type"] = "Bayesian/Reference";
   e["Problem"]["Constraints"][0] = 0;

   // Creating problem module
   Reference* p;
   knlohmann::json problemJs;
   problemJs["Type"] = "Bayesian/Reference";
   ASSERT_NO_THROW(p = dynamic_cast<Reference *>(Module::getModule(problemJs, &e)));
   e._problem = p;

   // Creating optimizer configuration Json
   knlohmann::json samplerJs;
   samplerJs["Type"] = "Sampler/TMCMC";
   samplerJs["Population Size"] = 512;

   // Creating module
   TMCMC* opt;
   ASSERT_NO_THROW(opt = dynamic_cast<TMCMC *>(Module::getModule(samplerJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(opt->applyModuleDefaults(samplerJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(opt->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = samplerJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   // Testing optional parameters
   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Current Burn In"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Current Burn In"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Pending Evaluation"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Pending Evaluation"] = std::vector<int>({0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Pending Gradient"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Pending Gradient"] = std::vector<int>({0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates LogLikelihoods"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates LogLikelihoods"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates LogPriors"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates LogPriors"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Gradients"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Gradients"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Errors"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Errors"] = std::vector<int>({0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Covariance"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Covariance"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders LogLikelihoods"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders LogLikelihoods"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders LogPriors"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders LogPriors"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders Gradients"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders Gradients"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders Errors"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders Errors"] = std::vector<int>({0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Finished Chains Count"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Finished Chains Count"] = 0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Current Chain Step"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Current Chain Step"] = std::vector<int>({0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Lengths"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Lengths"] = std::vector<int>({0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Coefficient Of Variation"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Coefficient Of Variation"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Count"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Count"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Annealing Exponent"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Annealing Exponent"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Previous Annealing Exponent"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Previous Annealing Exponent"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Finite Prior Evaluations"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Finite Prior Evaluations"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Accepted Samples Count"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Accepted Samples Count"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LogEvidence"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LogEvidence"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Proposals Acceptance Rate"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Proposals Acceptance Rate"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Selection Acceptance Rate"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Selection Acceptance Rate"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Covariance Matrix"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Covariance Matrix"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Loglikelihood"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Loglikelihood"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Mean Theta"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Mean Theta"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Database"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Database"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample LogLikelihood Database"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample LogLikelihood Database"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample LogPrior Database"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample LogPrior Database"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Gradient Database"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Gradient Database"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Error Database"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Error Database"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Covariances Database"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Covariances Database"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Upper Extended Boundaries"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Upper Extended Boundaries"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Lower Extended Boundaries"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Lower Extended Boundaries"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num LU Decomposition Failures Proposal"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num LU Decomposition Failures Proposal"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Eigen Decomposition Failures Proposal"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Eigen Decomposition Failures Proposal"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Inversion Failures Proposal"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Inversion Failures Proposal"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Negative Definite Proposals"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Negative Definite Proposals"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Cholesky Decomposition Failures Proposal"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Cholesky Decomposition Failures Proposal"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Covariance Corrections"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Covariance Corrections"] = 1;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   // Testing mandatory values

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Version"] = 1;
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Version"] = "Undefined";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Version");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Version"] = "TMCMC";
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Version"] = "mTMCMC";
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Population Size");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Population Size"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Population Size"] = 512;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Max Chain Length");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Chain Length"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Chain Length"] = 16;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Default Burn In");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Default Burn In"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Default Burn In"] = 4;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Per Generation Burn In");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Per Generation Burn In"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Per Generation Burn In"] = std::vector<size_t>({4});
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Target Coefficient Of Variation");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Target Coefficient Of Variation"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Target Coefficient Of Variation"] = 4;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Covariance Scaling");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Covariance Scaling"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Covariance Scaling"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Min Annealing Exponent Update");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Min Annealing Exponent Update"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Min Annealing Exponent Update"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Max Annealing Exponent Update");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Annealing Exponent Update"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Annealing Exponent Update"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Step Size");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Step Size"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Step Size"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Domain Extension Factor");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Domain Extension Factor"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Domain Extension Factor"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"].erase("Target Annealing Exponent");
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Target Annealing Exponent"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Target Annealing Exponent"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(samplerJs));
  }

} // namespace
