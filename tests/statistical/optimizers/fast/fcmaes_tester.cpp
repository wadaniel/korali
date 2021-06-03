#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fCMAES.hpp"

namespace
{

 float model(const std::vector<float>& x)
 {
  float result = 0.0f;
  for (size_t i = 0; i < x.size(); i++)
   result += -(x[i]+1)*(x[i]+1) + 2;
  return result;
 }

 using namespace korali;

 TEST(fCMAES, fullCovariance)
 {

  const size_t N = 4;
  const size_t popSize = 128;
  const size_t muSize = 32;
  const size_t seed = 1337;
  const float targetPrecision = 0.0001f;

  fCMAES c(N, popSize, muSize);
  ASSERT_NO_THROW(c.setSeed(seed));

  c._maxGenerations = 50;
  for (size_t i = 0; i < N; i++)
  {
   c._lowerBounds[i] = -10.0f;
   c._upperBounds[i] = +10.0f;
   c._initialMeans[i] = 0.0f;
   c._initialStandardDeviations[i] = 5.0f;
  }

  ASSERT_NO_THROW(c.reset());

  std::vector<float> candidateEvaluations(popSize);

  while (c.checkTermination() == false)
  {
   ASSERT_NO_THROW(c.prepareGeneration());

   for (size_t i = 0; i < popSize; i++)
    candidateEvaluations[i] = model(c._samplePopulation[i]);

   ASSERT_NO_THROW(c.updateDistribution(candidateEvaluations));
   c._currentGeneration++;
   c.printInfo();
  }

  for (size_t i = 0; i < N; i++)
   ASSERT_NEAR(c._bestEverVariables[i], -1.0f, 0.001);
  ASSERT_NEAR(c._bestEverValue, 8.0f, targetPrecision);
 }

 TEST(fCMAES, diagonalCovariance)
 {
  const size_t N = 4;
  const size_t popSize = 128;
  const size_t muSize = 32;
  const size_t seed = 1337;
  const float targetPrecision = 0.0001f;

  fCMAES c(N, popSize, muSize);
  ASSERT_NO_THROW(c.setSeed(seed));

  c._maxGenerations = 50;
  c._isDiagonal = 1;
  c._isSigmaBounded = 1;

  for (size_t i = 0; i < N; i++)
  {
   c._lowerBounds[i] = -10.0f;
   c._upperBounds[i] = +10.0f;
   c._initialMeans[i] = 0.0f;
   c._initialStandardDeviations[i] = 5.0f;
  }

  ASSERT_NO_THROW(c.reset());

  std::vector<float> candidateEvaluations(popSize);

  while (c.checkTermination() == false)
  {
   ASSERT_NO_THROW(c.prepareGeneration());

   for (size_t i = 0; i < popSize; i++)
    candidateEvaluations[i] = model(c._samplePopulation[i]);

   ASSERT_NO_THROW(c.updateDistribution(candidateEvaluations));
   c._currentGeneration++;
   c.printInfo();
  }

  for (size_t i = 0; i < N; i++)
   ASSERT_NEAR(c._bestEverVariables[i], -1.0f, 0.001);
  ASSERT_NEAR(c._bestEverValue, 8.0f, targetPrecision);
 }

} // namespace
