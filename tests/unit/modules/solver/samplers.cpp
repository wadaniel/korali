#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/sampler/sampler.hpp"
#include "modules/problem/sampling/sampling.hpp"
#include "modules/solver/sampler/MCMC/MCMC.hpp"

namespace
{
 using namespace korali;
 using namespace korali::solver;
 using namespace korali::solver::sampler;
 using namespace korali::problem;

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

} // namespace
