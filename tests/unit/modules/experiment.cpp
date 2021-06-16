#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/experiment/experiment.hpp"

namespace
{
 using namespace korali;

 TEST(Experiment, cornerCases)
 {
  Engine k;
  knlohmann::json expJs;
  expJs["Type"] = "Experiment";
  expJs["Problem"]["Type"] = "Optimization";
  expJs["Problem"]["Objective Function"] = 0;
  expJs["Solver"]["Type"] = "Optimizer/CMAES";
  expJs["Solver"]["Population Size"] = 16;
  expJs["Variables"][0]["Name"] = "Var 1";
  expJs["Variables"][0]["Lower Bound"] = -1.0;
  expJs["Variables"][0]["Upper Bound"] = 1.0;

  // Creating module
  Experiment* e;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  ASSERT_NO_THROW(e->applyModuleDefaults(expJs));
  ASSERT_NO_THROW(e->applyVariableDefaults());

  expJs["Type"] = "Experiment";
  auto backJs = expJs;
  e->_js.getJson() = expJs;

  e->_experimentId = 0;
  e->_engine = &k;
  e->_isFinished = false;
  e->initialize();

  expJs = backJs;
  expJs.erase("Variables");
  e->setConfiguration(expJs);


  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Current Generation"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Current Generation"] = 1;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Is Finished"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Is Finished"] = true;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Run ID"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Run ID"] = 1;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Timestamp"] = 1.0;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Timestamp"] = "00.00.00";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs.erase("Random Seed");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Random Seed"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Random Seed"] = 1;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs.erase("Distributions");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs.erase("Problem");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs.erase("Solver");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"].erase("Path");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Path"] = 1.0;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Path"] = "Path";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"].erase("Enabled");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Enabled"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Enabled"] = true;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"].erase("Frequency");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Frequency"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Frequency"] = 1;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs.erase("Store Sample Information");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Store Sample Information"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Store Sample Information"] = false;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Console Output"].erase("Verbosity");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Console Output"]["Verbosity"] = 1.0;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Console Output"]["Verbosity"] = "Silent";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Console Output"].erase("Frequency");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Console Output"]["Frequency"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Console Output"]["Frequency"] = 1;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));
 }

} // namespace
