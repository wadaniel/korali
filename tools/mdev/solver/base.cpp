#include "modules/solver/[MODULENAME]/[MODULENAME].hpp"
#include "modules/conduit/conduit.hpp"

void korali::solver::[CLASSNAME]::setInitialConfiguration()
{
 // Initialize solver's configuration here
}

void korali::solver::[CLASSNAME]::runGeneration()
{
 if (_k->_currentGeneration == 1) setInitialConfiguration();

 _k->_logger->logInfo("Normal", "Running generation %lu...\n", _k->_currentGeneration);

 // Example: Evaluating a sample
 korali::Sample sample;
 sample["Module"] = "Problem";
 sample["Operation"] = "Evaluate";
 korali::_conduit->start(sample);
 korali::_conduit->wait(sample);

 double evaluation = sample["Evaluation"];
 _k->_logger->logInfo("Normal", "Sample Evaluation: %f\n", evaluation);

}

void korali::solver::[CLASSNAME]::printGenerationBefore()
{
 _k->_logger->logInfo("Normal", "Preparing to start generation...\n");
}

void korali::solver::[CLASSNAME]::printGenerationAfter()
{
 _k->_logger->logInfo("Normal", "Finished to generation %lu...\n", _k->_currentGeneration);
}

