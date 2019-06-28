#include <stdio.h>

#include "korali.h"
#include "suite/testsuite.h"
#include "model/mvf.h"

using namespace Suite;

int main(int, char**)
{

  auto t = TestSuite(10, 1e-4); 
  
  t.addFactory("Prop", new CMAESFactory("Proportional", 0.7));
  //t.addFactory("Equal", new CMAESFactory("Equal", 0.7));
  //t.addFactory("Linear", new CMAESFactory("Linear", 0.7));
  //t.addFactory("Logarithmic", new CMAESFactory("Logarithmic", 0.7));
  //t.addFactory("DE Self Adaptive", new DEFactory("Best", "Self Adaptive", "Greedy"));
  //t.addFactory("DE Default", new DEFactory("Best", "Default", "Greedy"));

  
  //t.addTestFunction("Ackley", &mvfAckley, 64, -30.0, 30.0, 0.0, 100000);
  t.addTestFunction("Rosenbrock", &mvfRosenbrock, 2, -30.0, 30.0, 0.0, 100000);
  //t.addTestFunction("Beale", &mvfBeale, 4, -4.5, 4.5, 0.0, 100000);
  //t.addTestFunction("Bohachevsky1", &mvfBohachevsky1, 4, -50.0, 50.0, 0.0, 100000);
  
  t.run();
  t.makeStatistics();

  return 0;
}

