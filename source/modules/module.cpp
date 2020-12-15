#include "module.hpp"

#include "conduit/concurrent/concurrent.hpp"
#include "conduit/distributed/distributed.hpp"
#include "conduit/sequential/sequential.hpp"
#include "distribution/distribution.hpp"
#include "distribution/multivariate/normal/normal.hpp"
#include "distribution/specific/multinomial/multinomial.hpp"
#include "distribution/specific/specific.hpp"
#include "distribution/univariate/cauchy/cauchy.hpp"
#include "distribution/univariate/exponential/exponential.hpp"
#include "distribution/univariate/gamma/gamma.hpp"
#include "distribution/univariate/geometric/geometric.hpp"
#include "distribution/univariate/igamma/igamma.hpp"
#include "distribution/univariate/laplace/laplace.hpp"
#include "distribution/univariate/logNormal/logNormal.hpp"
#include "distribution/univariate/normal/normal.hpp"
#include "distribution/univariate/truncatedNormal/truncatedNormal.hpp"
#include "distribution/univariate/uniform/uniform.hpp"
#include "distribution/univariate/weibull/weibull.hpp"
#include "experiment/experiment.hpp"
#include "neuralNetwork/layer/feedforward/feedforward.hpp"
#include "neuralNetwork/layer/recurrent/recurrent.hpp"
#include "neuralNetwork/layer/input/input.hpp"
#include "neuralNetwork/layer/output/output.hpp"
#include "neuralNetwork/layer/layer.hpp"
#include "neuralNetwork/neuralNetwork.hpp"
#include "problem/bayesian/approximate/approximate.hpp"
#include "problem/bayesian/custom/custom.hpp"
#include "problem/bayesian/latent/exponentialLatent/exponentialLatent.hpp"
#include "problem/bayesian/latent/latent.hpp"
#include "problem/bayesian/reference/reference.hpp"
#include "problem/hierarchical/psi/psi.hpp"
#include "problem/hierarchical/theta/theta.hpp"
#include "problem/hierarchical/thetaNew/thetaNew.hpp"
#include "problem/integration/integration.hpp"
#include "problem/optimization/optimization.hpp"
#include "problem/problem.hpp"
#include "problem/propagation/propagation.hpp"
#include "problem/reinforcementLearning/continuous/continuous.hpp"
#include "problem/reinforcementLearning/discrete/discrete.hpp"
#include "problem/sampling/sampling.hpp"
#include "problem/supervisedLearning/supervisedLearning.hpp"
#include "solver/SAEM/SAEM.hpp"
#include "solver/agent/continuous/DDPG/DDPG.hpp"
#include "solver/agent/continuous/GFPT/GFPT.hpp"
#include "solver/agent/continuous/NAF/NAF.hpp"
#include "solver/agent/continuous/VRACER/VRACER.hpp"
#include "solver/agent/continuous/cACER/cACER.hpp"
#include "solver/agent/continuous/continuous.hpp"
#include "solver/agent/discrete/DDQN/DDQN.hpp"
#include "solver/agent/discrete/DQN/DQN.hpp"
#include "solver/agent/discrete/dACER/dACER.hpp"
#include "solver/agent/discrete/discrete.hpp"
#include "solver/executor/executor.hpp"
#include "solver/integrator/integrator.hpp"
#include "solver/learner/deepSupervisor/deepSupervisor.hpp"
#include "solver/learner/gaussianProcess/gaussianProcess.hpp"
#include "solver/optimizer/Adam/Adam.hpp"
#include "solver/optimizer/CMAES/CMAES.hpp"
#include "solver/optimizer/DEA/DEA.hpp"
#include "solver/optimizer/LMCMAES/LMCMAES.hpp"
#include "solver/optimizer/Rprop/Rprop.hpp"
#include "solver/optimizer/gridSearch/gridSearch.hpp"
#include "solver/optimizer/optimizer.hpp"
#include "solver/sampler/MCMC/MCMC.hpp"
#include "solver/sampler/Nested/Nested.hpp"
#include "solver/sampler/TMCMC/TMCMC.hpp"
#include "solver/sampler/sampler.hpp"

namespace korali
{
knlohmann::json __profiler;
std::chrono::time_point<std::chrono::high_resolution_clock> _startTime;
std::chrono::time_point<std::chrono::high_resolution_clock> _endTime;
double _cumulativeTime;

Module *Module::getModule(knlohmann::json &js, Experiment *e)
{
  std::string moduleType = "Undefined";

  if (!isDefined(js, "Type"))
    KORALI_LOG_ERROR(" + No module type provided in:\n %s\n", js.dump(2).c_str());

  try
  {
    moduleType = js["Type"].get<std::string>();
  }
  catch (const std::exception &ex)
  {
    KORALI_LOG_ERROR(" + Could not parse module type: '%s'.\n%s", js["Type"].dump(2).c_str(), ex.what());
  }

  moduleType.erase(remove_if(moduleType.begin(), moduleType.end(), isspace), moduleType.end());

  bool isExperiment = false;
  if (js["Type"] == "Experiment") isExperiment = true;

  // Once we've read the module type, we delete this information, because  it is not parsed by the module itself
  eraseValue(js, "Type");

  // Creating module pointer from it's type.
  Module *module = nullptr;

  if (moduleType == "Concurrent") module = new korali::conduit::Concurrent();
  if (moduleType == "Distributed") module = new korali::conduit::Distributed();
  if (moduleType == "Sequential") module = new korali::conduit::Sequential();
  if (moduleType == "Multivariate/Normal") module = new korali::distribution::multivariate::Normal();
  if (moduleType == "Specific/Multinomial") module = new korali::distribution::specific::Multinomial();
  if (moduleType == "Univariate/Cauchy") module = new korali::distribution::univariate::Cauchy();
  if (moduleType == "Univariate/Exponential") module = new korali::distribution::univariate::Exponential();
  if (moduleType == "Univariate/Gamma") module = new korali::distribution::univariate::Gamma();
  if (moduleType == "Univariate/Geometric") module = new korali::distribution::univariate::Geometric();
  if (moduleType == "Univariate/Igamma") module = new korali::distribution::univariate::Igamma();
  if (moduleType == "Univariate/Laplace") module = new korali::distribution::univariate::Laplace();
  if (moduleType == "Univariate/LogNormal") module = new korali::distribution::univariate::LogNormal();
  if (moduleType == "Univariate/Normal") module = new korali::distribution::univariate::Normal();
  if (moduleType == "Univariate/TruncatedNormal") module = new korali::distribution::univariate::TruncatedNormal();
  if (moduleType == "Univariate/Uniform") module = new korali::distribution::univariate::Uniform();
  if (moduleType == "Univariate/Weibull") module = new korali::distribution::univariate::Weibull();
  if (moduleType == "Experiment") module = new korali::Experiment();
  if (moduleType == "Bayesian/Approximate") module = new korali::problem::bayesian::Approximate();
  if (moduleType == "Bayesian/Custom") module = new korali::problem::bayesian::Custom();
  if (moduleType == "Bayesian/Latent") module = new korali::problem::bayesian::Latent();
  if (moduleType == "Bayesian/Latent/ExponentialLatent") module = new korali::problem::bayesian::latent::ExponentialLatent();
  if (moduleType == "Bayesian/Reference") module = new korali::problem::bayesian::Reference();
  if (moduleType == "Hierarchical/Psi") module = new korali::problem::hierarchical::Psi();
  if (moduleType == "Hierarchical/Theta") module = new korali::problem::hierarchical::Theta();
  if (moduleType == "Hierarchical/ThetaNew") module = new korali::problem::hierarchical::ThetaNew();
  if (moduleType == "Integration") module = new korali::problem::Integration();
  if (moduleType == "Optimization") module = new korali::problem::Optimization();
  if (moduleType == "Propagation") module = new korali::problem::Propagation();
  if (moduleType == "Sampling") module = new korali::problem::Sampling();
  if (moduleType == "ReinforcementLearning/Continuous") module = new korali::problem::reinforcementLearning::Continuous();
  if (moduleType == "ReinforcementLearning/Discrete") module = new korali::problem::reinforcementLearning::Discrete();
  if (moduleType == "SupervisedLearning") module = new korali::problem::SupervisedLearning();
  if (moduleType == "Executor") module = new korali::solver::Executor();
  if (moduleType == "Integrator") module = new korali::solver::Integrator();
  if (moduleType == "SAEM") module = new korali::solver::SAEM();
  if (moduleType == "Learner/GaussianProcess") module = new korali::solver::learner::GaussianProcess();
  if (moduleType == "Learner/DeepSupervisor") module = new korali::solver::learner::DeepSupervisor();
  if (moduleType == "Agent/Discrete/DACER") module = new korali::solver::agent::discrete::dACER();
  if (moduleType == "Agent/Discrete/DQN") module = new korali::solver::agent::discrete::DQN();
  if (moduleType == "Agent/Discrete/DDQN") module = new korali::solver::agent::discrete::DDQN();
  if (moduleType == "Agent/Continuous/CACER") module = new korali::solver::agent::continuous::cACER();
  if (moduleType == "Agent/Continuous/DDPG") module = new korali::solver::agent::continuous::DDPG();
  if (moduleType == "Agent/Continuous/GFPT") module = new korali::solver::agent::continuous::GFPT();
  if (moduleType == "Agent/Continuous/NAF") module = new korali::solver::agent::continuous::NAF();
  if (moduleType == "Agent/Continuous/VRACER") module = new korali::solver::agent::continuous::VRACER();
  if (moduleType == "Optimizer/CMAES") module = new korali::solver::optimizer::CMAES();
  if (moduleType == "Optimizer/DEA") module = new korali::solver::optimizer::DEA();
  if (moduleType == "Optimizer/Adam") module = new korali::solver::optimizer::Adam();
  if (moduleType == "Optimizer/Rprop") module = new korali::solver::optimizer::Rprop();
  if (moduleType == "Optimizer/LMCMAES") module = new korali::solver::optimizer::LMCMAES();
  if (moduleType == "Optimizer/GridSearch") module = new korali::solver::optimizer::GridSearch();
  if (moduleType == "Sampler/Nested") module = new korali::solver::sampler::Nested();
  if (moduleType == "Sampler/MCMC") module = new korali::solver::sampler::MCMC();
  if (moduleType == "Sampler/TMCMC") module = new korali::solver::sampler::TMCMC();
  if (moduleType == "NeuralNetwork") module = new korali::NeuralNetwork();
  if (moduleType == "Layer/FeedForward") module = new korali::neuralNetwork::layer::FeedForward();
  if (moduleType == "Layer/Recurrent") module = new korali::neuralNetwork::layer::Recurrent();
  if (moduleType == "Layer/Input") module = new korali::neuralNetwork::layer::Input();
  if (moduleType == "Layer/Output") module = new korali::neuralNetwork::layer::Output();

  if (module == nullptr) KORALI_LOG_ERROR(" + Unrecognized module: %s.\n", moduleType.c_str());

  // If this is a new experiment, we should assign it its own configuration
  if (isExperiment == true)
    dynamic_cast<Experiment *>(module)->_js.getJson() = js;

  // If this is a module inside an experiment, it needs to be properly configured
  if (isExperiment == false)
  {
    module->_k = e;
    module->applyVariableDefaults();
    module->applyModuleDefaults(js);
    module->setConfiguration(js);
  }

  return module;
}

Module *Module::duplicate(Module *src)
{
  knlohmann::json js;
  src->getConfiguration(js);
  return Module::getModule(js, src->_k);
}

} // namespace korali
