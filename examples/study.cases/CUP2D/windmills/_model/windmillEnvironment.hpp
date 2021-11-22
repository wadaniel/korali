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
std::vector<double> getUniformGridVort(Simulation *_environment, const std::array<Real,2> pos);
std::vector<double> getUniformLevelGridVort(Simulation *_environment, const std::array<Real,2> pos, int grid_level=3);

std::vector<double> vortGridProfile(Simulation *_environment);

std::vector<double> velGridProfile(Simulation *_environment);
std::vector<int> idRegion(const std::array<Real, 2> point, double height);

size_t holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo);
std::array<int, 2> safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh );