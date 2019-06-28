#include "testsuite.h"

#include <stdio.h>
#include <algorithm>

namespace Suite
{

TestSuite::TestSuite(size_t rep, double prec) : _repetitions(rep), _precision(prec)
{
  _factories.push_back(std::pair<std::string, EngineFactory*>("Prop", new CMAESFactory("Proportional", 0.7)));
  //_factories.push_back(std::pair<std::string, EngineFactory*>("Equal", new CMAESFactory("Equal", 0.7)));
  //_factories.push_back(std::pair<std::string, EngineFactory*>("Linear", new CMAESFactory("Linear", 0.7)));
  _factories.push_back(std::pair<std::string, EngineFactory*>("Logarithmic", new CMAESFactory("Logarithmic", 0.7)));
  //_factories.push_back(std::pair<std::string, EngineFactory*>("DE Self Adaptive", new DEFactory("Best", "Self Adaptive", "Greedy")));
  //_factories.push_back(std::pair<std::string, EngineFactory*>("DE Default", new DEFactory("Best", "Default", "Greedy")));
};

TestSuite::~TestSuite() {};

void TestSuite::run()
{


  for(auto func : _functions)
  {

    std::string funcname = func.first;

    std::map<std::string, std::vector<size_t>> evals;
   
    for(auto factory : _factories)
    {
    std::vector<size_t> evec;
 
    std::string fname = factory.first;
    
    for(size_t rep = 0; rep < _repetitions; ++rep)
    {

        auto k = factory.second->createEngine(_dimMap[funcname], _domainMap[funcname].first, _domainMap[funcname].second, _maxModelEvals[funcname], _fitnessMap[funcname] - _precision);

        auto model = [func](Korali::ModelData& d)
        {
          double res = func.second( d.getVariableCount(), &d.getVariables()[0] );
          d.addResult(res);
        };
        k.setModel(model);

        printf("\n\nRun Test: %s\nName: %s\n\n", funcname.c_str(), fname.c_str());

        k.run();
        auto config = k.getConfiguration();
        std::string solver = config["Solver"].get<std::string>();
        size_t eval = config[solver]["State"]["EvaluationCount"].get<size_t>();
        evec.push_back(eval);
    }

    
    evals.insert(std::pair<std::string, std::vector<size_t>>(fname, evec));
   }
    _evalsCount.insert(std::pair<std::string, std::map<std::string, std::vector<size_t>>>(funcname, evals));
  }
}


void TestSuite::addTestFunction(std::string name, TestFun fptr, size_t dim, double lowerBound, double upperBound, double fitness, size_t numFunEval )
{
  _functions.push_back( std::pair<std::string, TestFun> (name, fptr) );

  _dimMap.insert( std::pair<std::string, size_t>(name, dim) );

  std::pair<double, double> domain(lowerBound, upperBound);
  _domainMap.insert( std::pair<std::string, std::pair<double, double> > (name, domain ) );

  _fitnessMap.insert( std::pair<std::string, double>(name, fitness) );

  _maxModelEvals.insert( std::pair<std::string, size_t>(name, numFunEval) );
}

void TestSuite::addTargetFitness(std::string name, double fitness)
{
  _fitnessMap.insert( std::pair<std::string, double>(name, fitness) );
}


void TestSuite::addMaxFunctionEvaluations(std::string name, size_t numFunEval)
{
  _maxModelEvals.insert( std::pair<std::string, size_t>(name, numFunEval) );
}

void TestSuite::makeStatistics()
{
  for(auto func : _functions)
  {
    std::map<std::string, std::vector<size_t>> funcres = _evalsCount[func.first];
    for(auto fac : _factories)
    {
        std::vector<size_t> facres = funcres[fac.first];
        size_t min = *std::min_element(std::begin(facres), std::end(facres));
        size_t max = *std::max_element(std::begin(facres), std::end(facres));
        
        int sum = std::accumulate(facres.begin(), facres.end(), 0);
        double avg = (double)sum/facres.size();
        
        printf("| FUNCTION  %-25s  | SOLVER  %-25s | MIN  %zu | MAX %zu | AVG %f |\n", func.first.c_str(), fac.first.c_str(), min, max, avg); //TODO: sdev median
    }
  }

}

} // namespace Suite
