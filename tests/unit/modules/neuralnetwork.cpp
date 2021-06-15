#include "gtest/gtest.h"
#include "modules/neuralNetwork/neuralNetwork.hpp"
#include "modules/neuralNetwork/layer/layer.hpp"
#include "modules/neuralNetwork/layer/activation/activation.hpp"
#include "modules/neuralNetwork/layer/input/input.hpp"
#include "modules/neuralNetwork/layer/linear/linear.hpp"
#include "modules/neuralNetwork/layer/output/output.hpp"
#include "modules/neuralNetwork/layer/recurrent/recurrent.hpp"
#include "modules/neuralNetwork/layer/recurrent/gru/gru.hpp"
#include "modules/neuralNetwork/layer/recurrent/lstm/lstm.hpp"
#include "korali.hpp"

namespace
{
 using namespace korali;
 using namespace korali::neuralNetwork;
 using namespace korali::neuralNetwork::layer;
 using namespace korali::neuralNetwork::layer::recurrent;

 TEST(NeuralNetwork, BaseConfig)
 {
  Experiment e;
  e._logger = new Logger("Detailed", stdout);
  auto& experimentJs = e._js.getJson();

  NeuralNetwork* nn;
  knlohmann::json neuralNetworkConfig;
  neuralNetworkConfig["Type"] = "Neural Network";
  neuralNetworkConfig["Engine"] = "Korali";
  neuralNetworkConfig["Timestep Count"] = 1;
  neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
  neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
  neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
  neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Output";
  neuralNetworkConfig["Mode"] = "Training";

  ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
  ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
  auto baseNNJs = neuralNetworkConfig;

  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
  ASSERT_NO_THROW(nn->applyVariableDefaults());
  ASSERT_NO_THROW(nn->initialize());
  ASSERT_ANY_THROW(nn->initialize());

  std::vector<float> _hyperparameters;
  ASSERT_NO_THROW(_hyperparameters = nn->generateInitialHyperparameters());
  ASSERT_NO_THROW(nn->setHyperparameters(_hyperparameters));
  ASSERT_ANY_THROW(nn->setHyperparameters(std::vector<float>({0.1, 0.1})));

  nn->_isInitialized = false;
  nn->_engine = "CuDNN";
  ASSERT_NO_THROW(nn->initialize());

  nn->_isInitialized = false;
  nn->_batchSizes = std::vector<size_t>();
  ASSERT_ANY_THROW(nn->initialize());

  nn->_batchSizes = std::vector<size_t>({1});
  nn->_isInitialized = false;
  ASSERT_NO_THROW(nn->initialize());

  ASSERT_ANY_THROW(nn->forward({{{}}, {{}}}));
  ASSERT_ANY_THROW(nn->forward({{{}, {}}}));
  ASSERT_ANY_THROW(nn->forward({{{}}}));
  ASSERT_NO_THROW(nn->forward({{{0.0}}}));
  ASSERT_NO_THROW(nn->backward({{0.0}}));
  ASSERT_ANY_THROW(nn->backward({{0.0, 0.0}}));

  nn->_mode = "Inference";
  ASSERT_ANY_THROW(nn->backward({{0.0}}));
  nn->_mode = "Training";

  ASSERT_NO_THROW(nn->getInputGradients(1));
  ASSERT_ANY_THROW(nn->getInputGradients(0));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Current Training Loss"] = "Not a Number";
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Current Training Loss"] = 1.0;
  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig.erase("Engine");
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Engine"] = 1.0;
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Engine"] = "Korali";
  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig.erase("Mode");
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Mode"] = 1.0;
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Mode"] = "Training";
  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig.erase("Layers");
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Layers"] = knlohmann::json();
  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig.erase("Timestep Count");
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Timestep Count"] = "Not a Number";
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Timestep Count"] = 1;
  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig.erase("Batch Sizes");
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Batch Sizes"] = "Not a Number";
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Batch Sizes"] = std::vector<float>({1});
  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

  ASSERT_NO_THROW(nn->getConfiguration(neuralNetworkConfig));
 }

 TEST(NeuralNetwork, ActivationLayer)
  {
   Experiment e;
   e._logger = new Logger("Detailed", stdout);
   auto& experimentJs = e._js.getJson();

   NeuralNetwork* nn;
   knlohmann::json neuralNetworkConfig;
   neuralNetworkConfig["Type"] = "Neural Network";
   neuralNetworkConfig["Engine"] = "OneDNN";
   neuralNetworkConfig["Timestep Count"] = 1;
   neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
   neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
   neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
   neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Activation";
   neuralNetworkConfig["Layers"][1]["Function"] = "Elementwise/Linear";
   neuralNetworkConfig["Layers"][2]["Type"] = "Layer/Output";
   neuralNetworkConfig["Mode"] = "Training";

   ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
   ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
   auto baseNNJs = neuralNetworkConfig;

   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
   ASSERT_NO_THROW(nn->applyVariableDefaults());
   ASSERT_NO_THROW(nn->initialize());

   Activation* layer = dynamic_cast<Activation*>(nn->_pipelines[0][0]._layerVector[1]);
   ASSERT_NO_THROW(layer->applyVariableDefaults());
   knlohmann::json layerJs;
   ASSERT_NO_THROW(layer->getConfiguration(layerJs));

   layer->_function = "Elementwise/Clip";
   layer->_alpha = 0.3;
   layer->_beta = 0.6;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.5, 0.000001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/Clip";
   layer->_alpha = 0.3;
   layer->_beta = 0.4;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.4, 0.000001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/Linear";
   layer->_alpha = 2.0;
   layer->_beta = 0.6;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 1.6, 0.000001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/ReLU";
   layer->_alpha = 0.1;
   layer->_beta = 0.0;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.5, 0.001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/ReLU";
   layer->_alpha = 0.1;
   layer->_beta = 0.0;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{-0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], -0.05, 0.001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/Tanh";
   layer->_alpha = 0.0;
   layer->_beta = 0.0;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.46211715, 0.001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/Logistic";
   layer->_alpha = 0.0;
   layer->_beta = 0.0;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.62245933120, 0.001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/SoftSign";
   layer->_alpha = 0.0;
   layer->_beta = 0.0;
   ASSERT_ANY_THROW(layer->createForwardPipeline());

   layer->_function = "Softmax";
   layer->_alpha = 0.0;
   layer->_beta = 0.0;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 1.0, 0.001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   knlohmann::json baseLayerJs = layerJs;

   layerJs = baseLayerJs;
   layerJs.erase("Function");
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Function"] = 1.0;
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Function"] = "Unknown";
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Function"] = "Softmax";
   ASSERT_NO_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs.erase("Alpha");
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Alpha"] = "Not a Number";
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Alpha"] = 1.0;
   ASSERT_NO_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs.erase("Beta");
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Beta"] = "Not a Number";
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Beta"] = 1.0;
   ASSERT_NO_THROW(layer->setConfiguration(layerJs));
  }

} // namespace
