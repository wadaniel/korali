#include "korali.hpp"
#include <algorithm>
#include <random>
#include "Obstacles/Windmill.h"
#include "Simulation.h"
#include "Utils/BufferedLogger.h"

extern int _argc;
extern char **_argv;

void runEnvironment(korali::Sample &s);
void setInitialConditions(Windmill* agent, double init_angle, bool randomized);

std::vector<double> getConvState(Simulation *_environment, std::vector<double> center_area);
bool isInConvArea(const std::array<Real,2> point, std::vector<double> target, std::vector<double> dim);
std::vector<double> getUniformGridVort(Simulation *_environment);