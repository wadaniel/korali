#include <stdio.h>

#include "korali.h"
#include "suite/testsuite.h"
#include "model/mvf.h"

using namespace Suite;

int main(int, char**)
{

  auto t = TestSuite(10, 1e-4);
  
  //t.addTestFunction("Ackley", &mvfAckley, 64, -30.0, 30.0, 0.0, 100000);
  t.addTestFunction("Rosenbrock", &mvfRosenbrock, 32, -30.0, 30.0, 0.0, 100000);
  //t.addTestFunction("Beale", &mvfBeale, 4, -4.5, 4.5, 0.0, 100000);
  //t.addTestFunction("Bohachevsky1", &mvfBohachevsky1, 4, -50.0, 50.0, 0.0, 100000);
  
  t.run();
  t.makeStatistics();

  return 0;
}

