#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/learner/learner.hpp"
#include "modules/solver/learner/deepSupervisor/deepSupervisor.hpp"
#include "modules/solver/learner/gaussianProcess/gaussianProcess.hpp"
#include "modules/problem/supervisedLearning/supervisedLearning.hpp"

namespace
{
 using namespace korali;
 using namespace korali::solver;
 using namespace korali::solver::learner;
 using namespace korali::problem;

 //////////////// Deep Supervisor CLASS ////////////////////////

  TEST(learners, deepSupervisor)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();
   Variable v;
   e._variables.push_back(&v);

   // Creating optimizer configuration Json
   knlohmann::json learnerJs;

   // Using a neural network solver (deep learning) for inference

   learnerJs["Type"] = "Learner/DeepSupervisor";
   learnerJs["Loss Function"] = "Mean Squared Error";
   learnerJs["Steps Per Generation"] = 200;
   learnerJs["Learning Rate"] = 0.0001;

   // Defining the shape of the neural network

   learnerJs["Neural Network"]["Engine"] = "Korali";
   learnerJs["Neural Network"]["Optimizer"] = "Adam";

   learnerJs["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
   learnerJs["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32;

   learnerJs["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
   learnerJs["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

   learnerJs["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
   learnerJs["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32;

   learnerJs["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
   learnerJs["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";

   // Creating module
   DeepSupervisor* sampler;
   ASSERT_NO_THROW(sampler = dynamic_cast<DeepSupervisor *>(Module::getModule(learnerJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(sampler->applyModuleDefaults(learnerJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(sampler->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = learnerJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   // Testing optional parameters

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Current Loss"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Current Loss"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Normalization Means"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Normalization Means"] = std::vector<float>({1.0f});
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Normalization Variances"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Normalization Variances"] = std::vector<float>({1.0f});
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   // Testing mandatory parameters

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Neural Network"].erase("Hidden Layers");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Neural Network"]["Hidden Layers"] = knlohmann::json();
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Neural Network"].erase("Output Activation");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Neural Network"]["Output Activation"] = knlohmann::json();
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Neural Network"].erase("Output Layer");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Neural Network"]["Output Layer"] = knlohmann::json();
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Neural Network"].erase("Engine");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Neural Network"]["Engine"] = 1.0;
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Neural Network"]["Engine"] = "Engine";
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Neural Network"].erase("Optimizer");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Neural Network"]["Optimizer"] = 1.0;
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Neural Network"]["Optimizer"] = "Adam";
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs.erase("Hyperparameters");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Hyperparameters"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Hyperparameters"] = std::vector<float>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs.erase("Loss Function");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Loss Function"] = 1.0;
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Loss Function"] = "Direct Gradient";
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs.erase("Steps Per Generation");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Steps Per Generation"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Steps Per Generation"] = 10;
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs.erase("Learning Rate");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Learning Rate"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Learning Rate"] = 0.001;
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["L2 Regularization"].erase("Enabled");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["L2 Regularization"]["Enabled"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["L2 Regularization"]["Enabled"] = true;
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["L2 Regularization"].erase("Importance");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["L2 Regularization"]["Importance"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["L2 Regularization"]["Importance"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs.erase("Output Weights Scaling");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Output Weights Scaling"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Output Weights Scaling"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Termination Criteria"].erase("Target Loss");
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Termination Criteria"]["Target Loss"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(learnerJs));

   learnerJs = baseOptJs;
   experimentJs = baseExpJs;
   learnerJs["Termination Criteria"]["Target Loss"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(learnerJs));
  }

} // namespace
