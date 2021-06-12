#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/problem/reinforcementLearning/reinforcementLearning.hpp"
#include "modules/solver/agent/continuous/VRACER/VRACER.hpp"
#include "modules/solver/agent/discrete/dVRACER/dVRACER.hpp"


namespace
{
 using namespace korali;
 using namespace korali::solver;
 using namespace korali::solver::agent;
 using namespace korali::solver::agent::continuous;
 using namespace korali::solver::agent::discrete;
 using namespace korali::problem;

 //////////////// Base Agent CLASS ////////////////////////

  TEST(agent, baseAgent)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();
   experimentJs["Variables"][0]["Name"] = "X";
   Variable v;
   e._variables.push_back(&v);

   // Creating optimizer configuration Json
   knlohmann::json agentJs;

   // Using a neural network solver (deep learning) for inference

   agentJs["Type"] = "Agent / Continuous / VRACER";
   agentJs["Mode"] = "Training";
   agentJs["Episodes Per Generation"] = 10;
   agentJs["Experiences Between Policy Updates"] = 1;
   agentJs["Discount Factor"] = 0.99;
   agentJs["Learning Rate"] = 0.0001;
   agentJs["Mini Batch"]["Size"] = 32;
   agentJs["Experience Replay"]["Start Size"] = 1000;
   agentJs["Experience Replay"]["Maximum Size"] = 10000;

   /// Configuring the neural network and its hidden layers

   agentJs["Neural Network"]["Engine"] = "OneDNN";
   agentJs["Neural Network"]["Optimizer"] = "Adam";

   agentJs["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
   agentJs["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 16;

//   // Creating module
   VRACER* agent;
   ASSERT_NO_THROW(agent = dynamic_cast<VRACER *>(Module::getModule(agentJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(agent->applyModuleDefaults(agentJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(agent->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = agentJs;
   auto baseExpJs = experimentJs;

//   // Setting up optimizer correctly
   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   // Testing optional parameters
   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Action Lower Bounds"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Action Lower Bounds"] = std::vector<float>({0.0});
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Action Upper Bounds"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Action Upper Bounds"] = std::vector<float>({0.0});
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Current Episode"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Current Episode"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Training"]["Reward History"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Training"]["Reward History"] = std::vector<float>({1.0});
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Training"]["Experience History"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Training"]["Experience History"] = std::vector<float>({1});
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Training"]["Average Reward"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Training"]["Average Reward"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Training"]["Last Reward"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Training"]["Last Reward"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Training"]["Best Reward"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Training"]["Best Reward"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Training"]["Best Episode Id"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Training"]["Best Episode Id"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Reward"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Reward"] = std::vector<float>({1.0});
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Best Reward"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Best Reward"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Worst Reward"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Worst Reward"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Best Episode Id"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Best Episode Id"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Candidate Count"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Candidate Count"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Average Reward"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Average Reward"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Stdev Reward"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Stdev Reward"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Previous Average Reward"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Previous Average Reward"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Best Average Reward"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Best Average Reward"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["Count"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["Count"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["Ratio"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["Ratio"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["Current Cutoff"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["Current Cutoff"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Current Learning Rate"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Current Learning Rate"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy Update Count"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy Update Count"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Current Sample ID"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Current Sample ID"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Count"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Count"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Rescaling"]["Mean"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Rescaling"]["Mean"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Rescaling"]["Sigma"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Rescaling"]["Sigma"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Rescaling"]["Count"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Rescaling"]["Count"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Outbound Penalization"]["Count"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Outbound Penalization"]["Count"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["State Rescaling"]["Means"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["State Rescaling"]["Sigmas"] = std::vector<float>({1.0});
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   // Testing mandatory parameters

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs.erase("Mode");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Mode"] = 1.0;
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Mode"] = "Training";
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"].erase("Sample Ids");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Sample Ids"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Sample Ids"] = std::vector<size_t>({0});
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"].erase("Policy");

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Testing"]["Policy"] = std::vector<size_t>({0});
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs.erase("Agent Count");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Agent Count"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Agent Count"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs.erase("Episodes Per Generation");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Episodes Per Generation"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Episodes Per Generation"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Mini Batch"].erase("Size");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Mini Batch"]["Size"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Mini Batch"]["Size"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Mini Batch"].erase("Strategy");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Mini Batch"]["Strategy"] = 1;
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Mini Batch"]["Strategy"] = "Uniform";
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs.erase("Time Sequence Length");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Time Sequence Length"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Time Sequence Length"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs.erase("Learning Rate");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Learning Rate"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Learning Rate"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["L2 Regularization"].erase("Enabled");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["L2 Regularization"]["Enabled"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["L2 Regularization"]["Enabled"] = true;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["L2 Regularization"].erase("Importance");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["L2 Regularization"]["Importance"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["L2 Regularization"]["Importance"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Neural Network"].erase("Hidden Layers");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Neural Network"]["Hidden Layers"] = knlohmann::json();
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Neural Network"].erase("Optimizer");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Neural Network"]["Optimizer"] = 1.0;
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Neural Network"]["Optimizer"] = "Adam";
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Neural Network"].erase("Engine");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Neural Network"]["Engine"] = 1.0;
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Neural Network"]["Engine"] = "Adam";
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs.erase("Discount Factor");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Discount Factor"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Discount Factor"] = 0.99;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"].erase("Serialize");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Serialize"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Serialize"] = true;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"].erase("Start Size");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Start Size"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Start Size"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"].erase("Maximum Size");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Maximum Size"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Maximum Size"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"].erase("Cutoff Scale");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["Cutoff Scale"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"].erase("Target");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["Target"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["Target"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"].erase("Annealing Rate");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["Annealing Rate"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["Annealing Rate"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"].erase("REFER Beta");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["REFER Beta"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experience Replay"]["Off Policy"]["REFER Beta"] = 1.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs.erase("Experiences Between Policy Updates");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experiences Between Policy Updates"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Experiences Between Policy Updates"] = 1;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["State Rescaling"].erase("Enabled");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["State Rescaling"]["Enabled"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["State Rescaling"]["Enabled"] = true;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Outbound Penalization"].erase("Enabled");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Outbound Penalization"]["Enabled"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Outbound Penalization"]["Enabled"] = true;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Outbound Penalization"].erase("Factor");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Outbound Penalization"]["Factor"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Reward"]["Outbound Penalization"]["Factor"] = 2.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"].erase("Max Experiences");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"]["Max Experiences"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"]["Max Experiences"] = 200;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"].erase("Max Episodes");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"]["Max Episodes"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"]["Max Episodes"] = 200;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"]["Testing"].erase("Target Average Reward");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"]["Testing"]["Target Average Reward"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"]["Testing"]["Target Average Reward"]  = 200.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"]["Testing"].erase("Average Reward Increment");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"]["Testing"]["Average Reward Increment"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"]["Testing"]["Average Reward Increment"]  = 200.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"].erase("Max Policy Updates");
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"]["Max Policy Updates"] = "Not a Number";
   ASSERT_ANY_THROW(agent->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Termination Criteria"]["Max Policy Updates"]  = 200.0;
   ASSERT_NO_THROW(agent->setConfiguration(agentJs));
  }

} // namespace
