#include <stdio.h>

#include "korali.h"
#include "suite/testsuite.h"
#include "model/mvf.h"

using namespace Suite;

int main(int, char**)
{
    auto t = TestSuite();
    t.addTestFunction("Ackley", &mvfAckley);

    t.run();

    return 0;
}

