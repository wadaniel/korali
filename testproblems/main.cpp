#include <stdio.h>

#include "korali.h"
#include "suite/testsuite.h"
#include "model/mvf.h"

using namespace Suite;

int main(int, char**)
{

  auto t = TestSuite();
  
  t.addTestFunction("Ackley", &mvfAckley, 4, -30.0, 30.0, 0.0, 100000);
  t.addTestFunction("Beale", &mvfBeale, 2, -4.5, 4.5, 0.0, 100000);
  t.addTestFunction("Bohachevsky1", &mvfBohachevsky1, 2, -50.0, 50.0, 0.0, 100000);
  
  t.run();

  return 0;
}

