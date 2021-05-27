#include "engine.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/solver/sampler/Nested/Nested.hpp"
#include "sample/sample.hpp"

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_eigen.h>

#include <algorithm> //sort
#include <chrono>
#include <limits>
#include <math.h> //isfinite, sqrt
#include <numeric>
#include <random> // std::default_random_engine

#define L2DIST

namespace korali
{
namespace solver
{
namespace sampler
{


void ellipse_t::initSphere()
{
  num = 0;
  sampleIdx.clear();
  std::fill(mean.begin(), mean.end(), 0.0);
  std::fill(cov.begin(), cov.end(), 0.0);
  std::fill(invCov.begin(), invCov.end(), 0.0);
  std::fill(axes.begin(), axes.end(), 0.0);
  std::fill(evals.begin(), evals.end(), 0.0);
  std::fill(paxes.begin(), paxes.end(), 0.0);

  for (size_t d = 0; d < dim; ++d)
  {
    cov[d * dim + d] = 1.0;
    invCov[d * dim + d] = 1.0;
    axes[d * dim + d] = 1.0;
    evals[d] = 1.0;
    paxes[d * dim + d] = 1.0;
  }
  det = 1.0;
  volume = sqrt(pow(M_PI, dim)) * 2.0 / ((double)dim * gsl_sf_gamma(0.5 * dim));
  pointVolume = 0.0;
}

void ellipse_t::scaleVolume(double factor)
{
  double K = sqrt(pow(M_PI, dim)) * 2.0 / ((double)dim * gsl_sf_gamma(0.5 * dim));
  double enlargementFactor = pow((volume * volume * factor * factor) / (K * K * det), 1.0 / ((double)dim));
  for (size_t d = 0; d < dim * dim; ++d)
  {
    cov[d] *= enlargementFactor;
    invCov[d] *= 1.0 / enlargementFactor;
    axes[d] *= enlargementFactor;
  }
  for (size_t d = 0; d < dim; ++d) evals[d] *= enlargementFactor;
}

void Nested::setInitialConfiguration()
{
  _shuffleSeed = _k->_randomSeed++;
  _variableCount = _k->_variables.size();

  if (_minLogEvidenceDelta < 0.0) KORALI_LOG_ERROR("Min Log Evidence Delta must be larger equal 0.0 (is %lf).\n", _minLogEvidenceDelta);

  if ((_resamplingMethod != "Box") && (_resamplingMethod != "Ellipse") && (_resamplingMethod != "Multi Ellipse")) KORALI_LOG_ERROR("Only accepted Resampling Method are 'Box', 'Ellipse' and 'Multi Ellipse' (is %s).\n", _resamplingMethod.c_str());

  if (_proposalUpdateFrequency <= 0) KORALI_LOG_ERROR("Proposal Update Frequency must be larger 0");

  _priorLowerBound.resize(_variableCount);
  _priorWidth.resize(_variableCount);

  for (size_t d = 0; d < _variableCount; ++d)
  {
    if (dynamic_cast<distribution::Univariate *>(_k->_distributions[_k->_variables[d]->_distributionIndex]) == nullptr) KORALI_LOG_ERROR("Prior of variable %s is not of type Univariate (is %s).\n", _k->_variables[d]->_name.c_str(), _k->_distributions[_k->_variables[d]->_distributionIndex]->_type.c_str());

    if ((iCompare(_k->_distributions[_k->_variables[d]->_distributionIndex]->_type, "Univariate/Uniform") == false) && (std::isfinite(_k->_variables[d]->_lowerBound) == false)) KORALI_LOG_ERROR("Prior of variable %s is not 'Univariate/Uniform' (is %s) AND lower bound not set (invalid configuration).\n", _k->_variables[d]->_name.c_str(), _k->_distributions[_k->_variables[d]->_distributionIndex]->_type.c_str());

    if ((iCompare(_k->_distributions[_k->_variables[d]->_distributionIndex]->_type, "Univariate/Uniform") == false) && (std::isfinite(_k->_variables[d]->_upperBound) == false)) KORALI_LOG_ERROR("Prior of variable %s is not 'Univariate/Uniform' (is %s) AND upper bound not set (invalid configuration).\n", _k->_variables[d]->_name.c_str(), _k->_distributions[_k->_variables[d]->_distributionIndex]->_type.c_str());

    if (iCompare(_k->_distributions[_k->_variables[d]->_distributionIndex]->_type, "Univariate/Uniform"))
    {
      _priorWidth[d] = dynamic_cast<distribution::univariate::Uniform *>(_k->_distributions[_k->_variables[d]->_distributionIndex])->_maximum - dynamic_cast<distribution::univariate::Uniform *>(_k->_distributions[_k->_variables[d]->_distributionIndex])->_minimum;
      _priorLowerBound[d] = dynamic_cast<distribution::univariate::Uniform *>(_k->_distributions[_k->_variables[d]->_distributionIndex])->_minimum;
    }
    else
    {
      _priorWidth[d] = _k->_variables[d]->_upperBound - _k->_variables[d]->_lowerBound;
      _priorLowerBound[d] = _k->_variables[d]->_lowerBound;
    }
  }

  if ((_resamplingMethod == "Ellipse" || _resamplingMethod == "Multi Ellipse") && (_variableCount < 3)) KORALI_LOG_ERROR("Resampling Method 'Ellipse' and 'Multi Ellipse' only suitable for problems of dim larger 2 (use Resampling Method 'Box').");

  _candidateLogLikelihoods.resize(_batchSize);
  _candidateLogPriors.resize(_batchSize);
  _candidates.resize(_batchSize);
  for (size_t i = 0; i < _batchSize; i++) _candidates[i].resize(_variableCount);

  _liveLogLikelihoods.resize(_numberLivePoints);
  _liveLogPriors.resize(_numberLivePoints);
  _liveSamplesRank.resize(_numberLivePoints);
  _liveSamples.resize(_numberLivePoints);
  for (size_t i = 0; i < _numberLivePoints; i++) _liveSamples[i].resize(_variableCount);

  _databaseEntries = 0;
  _sampleLogLikelihoodDatabase.resize(0);
  _sampleLogPriorDatabase.resize(0);
  _sampleLogWeightDatabase.resize(0);
  _sampleDatabase.resize(0);

  // Init Generation
  _logEvidence = std::numeric_limits<double>::lowest();
  _sumLogWeights = std::numeric_limits<double>::lowest();
  _sumSquareLogWeights = std::numeric_limits<double>::lowest();
  _logEvidenceDifference = std::numeric_limits<double>::max();
  _expectedLogShrinkage = log((_numberLivePoints + 1.) / _numberLivePoints);
  _logVolume = 0;

  _logEvidenceVar = 0.;
  _information = 0.;
  _lastAccepted = 0;
  _nextUpdate = 0;
  _acceptedSamples = 0;
  _generatedSamples = 0;
  _lStarOld = std::numeric_limits<double>::lowest();
  _lStar = std::numeric_limits<double>::lowest();

  _domainMean.resize(_variableCount);
  if (_resamplingMethod == "Box")
  {
    _boxLowerBound.resize(_variableCount);
    _boxUpperBound.resize(_variableCount);
  }
  else if (_resamplingMethod == "Ellipse")
  {
    initEllipseVector();
  }
  else /* _resamplingMethod == "Multi Ellipse" */
  {
    initEllipseVector();
  }

  (*_k)["Results"]["Posterior Samples"] = {};
}

void Nested::runGeneration()
{
  if (_k->_currentGeneration == 1)
  {
    setInitialConfiguration();
    runFirstGeneration();
    return;
  };

  // Generation > 1
  bool accepted;
  _lastAccepted = 0;
  std::vector<double> sample;
  std::vector<Sample> samples(_batchSize);

  do
  {
    updateBounds();
    generateCandidates();

    for (size_t c = 0; c < _batchSize; c++)
    {
      samples[c]["Module"] = "Problem";
      samples[c]["Operation"] = "Evaluate";
      sample = _candidates[c];
      priorTransform(sample);
      samples[c]["Parameters"] = sample;
      samples[c]["Sample Id"] = c;
      KORALI_START(samples[c]);
      _modelEvaluationCount++;
      _generatedSamples++;
    }

    size_t finishedCandidatesCount = 0;
    while (finishedCandidatesCount < _batchSize)
    {
      size_t finishedId = KORALI_WAITANY(samples);

      auto candidate = KORALI_GET(std::vector<double>, samples[finishedId], "Parameters");
      _candidateLogPriors[finishedId] = KORALI_GET(double, samples[finishedId], "logPrior");
      _candidateLogLikelihoods[finishedId] = KORALI_GET(double, samples[finishedId], "logLikelihood");
      _candidateLogLikelihoods[finishedId] += logPriorWeight(candidate);

      finishedCandidatesCount++;
    }

    _lastAccepted++;
    accepted = processGeneration();

  } while (accepted == false);

  return;
}

void Nested::runFirstGeneration()
{
  for (size_t i = 0; i < _numberLivePoints; i++)
    for (size_t d = 0; d < _variableCount; d++)
      _liveSamples[i][d] = _uniformGenerator->getRandomNumber();

  std::vector<double> sample;
  std::vector<Sample> samples(_numberLivePoints);

  for (size_t c = 0; c < _numberLivePoints; c++)
  {
    samples[c]["Module"] = "Problem";
    samples[c]["Operation"] = "Evaluate";
    sample = _liveSamples[c];
    priorTransform(sample);
    samples[c]["Parameters"] = sample;
    samples[c]["Sample Id"] = c;
    KORALI_START(samples[c]);
    _modelEvaluationCount++;
    _generatedSamples++;
  }

  size_t finishedCandidatesCount = 0;
  while (finishedCandidatesCount < _numberLivePoints)
  {
    size_t finishedId = KORALI_WAITANY(samples);

    auto sample = KORALI_GET(std::vector<double>, samples[finishedId], "Parameters");
    _liveLogPriors[finishedId] = KORALI_GET(double, samples[finishedId], "logPrior");
    _liveLogLikelihoods[finishedId] = KORALI_GET(double, samples[finishedId], "logLikelihood");
    _liveLogLikelihoods[finishedId] += logPriorWeight(sample);

    finishedCandidatesCount++;
  }

  sortLiveSamplesAscending();

  if (isfinite(_liveLogLikelihoods[_liveSamplesRank[0]])) _lStar = _liveLogLikelihoods[_liveSamplesRank[0]];
  _maxEvaluation = _liveLogLikelihoods[_liveSamplesRank[_numberLivePoints - 1]];

  return;
}

void Nested::updateBounds()
{
  if (_generatedSamples < _nextUpdate) return;

  _nextUpdate += _proposalUpdateFrequency;

  if (_resamplingMethod == "Box")
  {
    updateBox();
  }
  else if (_resamplingMethod == "Ellipse")
  {
    updateEllipse(_ellipseVector.front());
  }
  else /* _resamplingMethod == "Multi Ellipse" */
  {
    updateMultiEllipse();
  }
}

void Nested::priorTransform(std::vector<double> &sample) const
{
  for (size_t d = 0; d < _variableCount; ++d) sample[d] = _priorLowerBound[d] + sample[d] * _priorWidth[d];
}

void Nested::generateCandidates()
{
  if (_resamplingMethod == "Box")
  {
    generateCandidatesFromBox();
  }
  else if (_resamplingMethod == "Ellipse")
  {
    generateCandidatesFromEllipse();
  }
  else /* _resamplingMethod == "Multi Ellipse" */
  {
    generateCandidatesFromMultiEllipse();
  }
}

bool Nested::processGeneration()
{
  size_t sampleIdx = _liveSamplesRank[0];
  size_t acceptedBefore = _acceptedSamples;
  for (size_t c = 0; c < _batchSize; ++c)
  {
    if (_candidateLogLikelihoods[c] < _lStar) continue;
    _acceptedSamples++;

    // update evidence & domain
    double logVolumeOld = _logVolume;
    double informationOld = _information;
    double logEvidenceOld = _logEvidence;

    _logVolume -= _expectedLogShrinkage;

    double dLogVol = log(0.5 * exp(logVolumeOld) - 0.5 * exp(_logVolume));
    _logWeight = safeLogPlus(_lStar, _lStarOld) + dLogVol;
    _logEvidence = safeLogPlus(_logEvidence, _logWeight);

    double evidenceTerm = exp(_lStarOld - _logEvidence) * _lStarOld + exp(_lStar - _logEvidence) * _lStar;

    if (isfinite(evidenceTerm))
    {
      _information = exp(dLogVol) * evidenceTerm + exp(logEvidenceOld - _logEvidence) * (informationOld + logEvidenceOld) - _logEvidence;
      _logEvidenceVar += 2. * (_information - informationOld) * _expectedLogShrinkage;
    }

    // add it to db
    if (isfinite(_liveLogLikelihoods[sampleIdx])) updateSampleDatabase(sampleIdx);

    // replace worst sample
    _liveSamples[sampleIdx] = _candidates[c];
    _liveLogPriors[sampleIdx] = _candidateLogPriors[c];
    _liveLogLikelihoods[sampleIdx] = _candidateLogLikelihoods[c];

    // sort rank vector and update constraint
    sortLiveSamplesAscending();

    // select new worst sample
    sampleIdx = _liveSamplesRank[0];

    _lStarOld = _lStar;
    if (isfinite(_liveLogLikelihoods[sampleIdx])) _lStar = _liveLogLikelihoods[sampleIdx];
  }

  _maxEvaluation = _liveLogLikelihoods[_liveSamplesRank[_numberLivePoints - 1]];
  _remainingLogEvidence = _maxEvaluation + _logVolume;
  _logEvidenceDifference = safeLogPlus(_logEvidence, _remainingLogEvidence) - _logEvidence;
  setBoundsVolume();

  _lastAccepted++;
  return (acceptedBefore != _acceptedSamples);
}

double Nested::logPriorWeight(std::vector<double> &sample)
{
  double logweight = 0.0;
  for (size_t d = 0; d < _variableCount; ++d)
    if (_k->_distributions[_k->_variables[d]->_distributionIndex]->_type != "Univariate/Uniform")
    {
      logweight += dynamic_cast<distribution::Univariate *>(_k->_distributions[_k->_variables[d]->_distributionIndex])->getLogDensity(sample[d]);
      logweight += log(_priorWidth[d]);
    }
  return logweight;
}

void Nested::setBoundsVolume()
{
  if (_resamplingMethod == "Box")
  {
    _boundLogVolume = std::numeric_limits<double>::lowest();
    for (size_t d = 0; d < _variableCount; ++d)
      _boundLogVolume = safeLogPlus(_boundLogVolume, log(_boxUpperBound[d] - _boxLowerBound[d]));
  }
  else if (_resamplingMethod == "Ellipse")
  {
    auto &ellipse = _ellipseVector.front();
    _boundLogVolume = log(ellipse.volume);
  }
  else /* _resamplingMethod == "Multi Ellipse" */
  {
    _boundLogVolume = std::numeric_limits<double>::lowest();
    for (auto &ellipse : _ellipseVector)
      _boundLogVolume = safeLogPlus(_boundLogVolume, log(ellipse.volume));
  }
}

void Nested::generateCandidatesFromBox()
{
  for (size_t i = 0; i < _batchSize; i++)
  {
    for (size_t d = 0; d < _variableCount; ++d)
      _candidates[i][d] = _boxLowerBound[d] + _uniformGenerator->getRandomNumber() * (_boxUpperBound[d] - _boxLowerBound[d]);
    if (insideUnitCube(_candidates[i]) == false) i--;
  }
}

void Nested::generateSampleFromEllipse(const ellipse_t &ellipse, std::vector<double> &sample) const
{
  double len = 0;
  std::vector<double> vec(_variableCount);
  for (size_t d = 0; d < _variableCount; ++d)
  {
    vec[d] = _normalGenerator->getRandomNumber();
    len += vec[d] * vec[d];
  }
  for (size_t d = 0; d < _variableCount; ++d)
    vec[d] *= pow(_uniformGenerator->getRandomNumber(), 1. / ((double)_variableCount)) / sqrt(len);

  for (size_t k = 0; k < _variableCount; ++k)
  {
    sample[k] = ellipse.mean[k];
    for (size_t l = 0; l < k + 1; ++l)
    {
      sample[k] += ellipse.axes[k * _variableCount + l] * vec[l];
    }
  }
}

void Nested::generateCandidatesFromEllipse()
{
  for (size_t i = 0; i < _batchSize; i++)
  {
    generateSampleFromEllipse(_ellipseVector.front(), _candidates[i]);
    if (insideUnitCube(_candidates[i]) == false) i--;
  }
}

void Nested::generateCandidatesFromMultiEllipse()
{
  double totalVol = 0.0;
  for (auto &ellipse : _ellipseVector) totalVol += ellipse.volume;

  for (size_t i = 0; i < _batchSize; i++)
  {
    // randomly select ellipse
    double cumVol = 0.0;
    double rnd_ellipse = _uniformGenerator->getRandomNumber() * totalVol;

    ellipse_t *ellipse_ptr = NULL;
    for (auto &ellipse : _ellipseVector)
    {
      cumVol += ellipse.volume;
      if (rnd_ellipse < cumVol)
      {
        ellipse_ptr = &ellipse;
        break;
      }
    }

    if (ellipse_ptr == NULL)
      KORALI_LOG_ERROR("Failed to assign ellipse vector.");

    bool accept = false;
    while (accept == false)
    {
      // sample from ellipse
      generateSampleFromEllipse(*ellipse_ptr, _candidates[i]);

      // check for overlaps
      size_t overlap = 0;
      for (auto &ellipse : _ellipseVector)
      {
        double dist = mahalanobisDistance(_candidates[i], ellipse);
        if (dist <= 1.0) overlap++;
      }

      // accept / reject
      double rnd = _uniformGenerator->getRandomNumber();
      if (rnd < 1.0 / ((double)overlap)) accept = true;
      if (insideUnitCube(_candidates[i]) == false) accept = false;
    }
  }
}

void Nested::updateBox()
{
  for (size_t d = 0; d < _variableCount; d++) _boxLowerBound[d] = std::numeric_limits<double>::max();
  for (size_t d = 0; d < _variableCount; d++) _boxUpperBound[d] = std::numeric_limits<double>::lowest();

  for (size_t i = 0; i < _numberLivePoints; i++)
    for (size_t d = 0; d < _variableCount; d++)
    {
      _boxLowerBound[d] = std::min(_boxLowerBound[d], _liveSamples[i][d]);
      _boxUpperBound[d] = std::max(_boxUpperBound[d], _liveSamples[i][d]);
    }
}

void Nested::sortLiveSamplesAscending()
{
  std::iota(_liveSamplesRank.begin(), _liveSamplesRank.end(), 0);
  sort(_liveSamplesRank.begin(), _liveSamplesRank.end(), [this](const size_t &idx1, const size_t &idx2) -> bool { return this->_liveLogLikelihoods[idx1] < this->_liveLogLikelihoods[idx2]; });
}

void Nested::updateSampleDatabase(size_t sampleIdx)
{
  _databaseEntries++;
  _sampleDatabase.push_back(_liveSamples[sampleIdx]);
  priorTransform(_sampleDatabase.back());

  _sampleLogPriorDatabase.push_back(_liveLogPriors[sampleIdx]);
  _sampleLogLikelihoodDatabase.push_back(_liveLogLikelihoods[sampleIdx]);
  _sampleLogWeightDatabase.push_back(_logWeight);

  updateEffectiveSamples();
}

void Nested::consumeLiveSamples()
{
  size_t sampleIdx;
  double dLogVol, logEvidenceOld, informationOld, evidenceTerm;

  std::vector<double> logvols(_numberLivePoints + 1, _logVolume);
  std::vector<double> logdvols(_numberLivePoints);
  std::vector<double> dlvs(_numberLivePoints);

  for (size_t i = 0; i < _numberLivePoints; ++i)
  {
    logvols[i + 1] += log(1. - (i + 1.) / (_numberLivePoints + 1.));
    logdvols[i] = safeLogMinus(logvols[i], logvols[i + 1]);
    dlvs[i] = logvols[i] - logvols[i + 1];
  }
  for (size_t i = 0; i < _numberLivePoints + 1; ++i) logdvols[i] += log(0.5);

  for (size_t i = 0; i < _numberLivePoints; ++i)
  {
    sampleIdx = _liveSamplesRank[i];

    logEvidenceOld = _logEvidence;
    informationOld = _information;

    _lStarOld = _lStar;
    if (isfinite(_liveLogLikelihoods[sampleIdx])) _lStar = _liveLogLikelihoods[sampleIdx];
    dLogVol = logdvols[i];

    _logVolume = safeLogMinus(_logVolume, dLogVol);
    _logWeight = safeLogPlus(_lStar, _lStarOld) + dLogVol;
    _logEvidence = safeLogPlus(_logEvidence, _logWeight);

    evidenceTerm = exp(_lStarOld - _logEvidence) * _lStarOld + exp(_lStar - _logEvidence) * _lStar;

    _information = exp(dLogVol) * evidenceTerm + exp(logEvidenceOld - _logEvidence) * (informationOld + logEvidenceOld) - _logEvidence;

    _logEvidenceVar += 2. * (_information - informationOld) * dlvs[i];

    if (isfinite(_liveLogLikelihoods[sampleIdx])) updateSampleDatabase(sampleIdx);
  }
}

void Nested::generatePosterior()
{
  double maxLogWtDb = *max_element(std::begin(_sampleLogWeightDatabase), std::end(_sampleLogWeightDatabase));

  std::vector<size_t> permutation(_databaseEntries);
  std::iota(std::begin(permutation), std::end(permutation), 0);
  std::shuffle(permutation.begin(), permutation.end(), std::default_random_engine(_shuffleSeed));

  size_t rndIdx;
  std::vector<std::vector<double>> posteriorSamples;
  std::vector<double> posteriorSampleLogPriorDatabase;
  std::vector<double> posteriorSampleLogLikelihoodDatabase;

  double k = 1.0;
  double sum = _uniformGenerator->getRandomNumber();
  for (size_t i = 0; i < _databaseEntries; ++i)
  {
    rndIdx = permutation[i];
    sum += exp(_sampleLogWeightDatabase[rndIdx] - maxLogWtDb);
    if (sum > k)
    {
      posteriorSamples.push_back(_sampleDatabase[rndIdx]);
      posteriorSampleLogPriorDatabase.push_back(_sampleLogPriorDatabase[rndIdx]);
      posteriorSampleLogLikelihoodDatabase.push_back(_sampleLogLikelihoodDatabase[rndIdx]);
      k++;
    }
  }

  (*_k)["Results"]["Posterior Sample Database"] = posteriorSamples;
  (*_k)["Results"]["Posterior Sample LogPrior Database"] = posteriorSampleLogPriorDatabase;
  (*_k)["Results"]["Posterior Sample LogLikelihood Database"] = posteriorSampleLogLikelihoodDatabase;
}

double Nested::l2distance(const std::vector<double> &sampleOne, const std::vector<double> &sampleTwo) const
{
  double dist = 0.;
  for (size_t d = 0; d < _variableCount; ++d) dist += (sampleOne[d] - sampleTwo[d]) * (sampleOne[d] - sampleTwo[d]);
  dist = sqrt(dist);
  return dist;
}

bool Nested::updateEllipse(ellipse_t &ellipse) const
{
  if (ellipse.num == 0) return false;

  updateEllipseMean(ellipse);
  bool good = updateEllipseCov(ellipse);
  if (good) good = updateEllipseVolume(ellipse);

  return good;
}

void Nested::updateMultiEllipse()
{
  initEllipseVector();
  bool ok = updateEllipse(_ellipseVector.front());
  if (ok == false) KORALI_LOG_ERROR("Ellipse update failed at initialization\n");

  bool okCluster, okOne, okTwo;

  std::vector<ellipse_t> newEllipseVector;
  for (size_t idx = 0; idx < _ellipseVector.size(); ++idx)
  {
    ellipse_t &ellipse = _ellipseVector[idx];
    auto one = ellipse_t(_variableCount);
    auto two = ellipse_t(_variableCount);

    okCluster = kmeansClustering(ellipse, 100, one, two);

#ifdef WMDIST
    if (okCluster)
    {
      okOne = updateEllipseVolume(one);
      okTwo = updateEllipseVolume(two);
    }
#else
    if (okCluster)
    {
      okOne = updateEllipse(one);
      okTwo = updateEllipse(two);
    }
#endif

    if ((okCluster == false) || (okOne == false) || (okTwo == false) || ((one.volume + two.volume >= 0.5 * ellipse.volume) && (ellipse.volume < 2.0 * exp(_logVolume))))
    {
      newEllipseVector.push_back(ellipse);
    }
    else
    {
      _ellipseVector.push_back(one);
      _ellipseVector.push_back(two);
    }
  }

  _ellipseVector = newEllipseVector;
}

void Nested::initEllipseVector()
{
  _ellipseVector.clear();
  _ellipseVector.emplace_back(ellipse_t(_variableCount));
  ellipse_t *first = _ellipseVector.data();

  first->num = _numberLivePoints;
  first->sampleIdx.resize(_numberLivePoints);

  std::iota(first->sampleIdx.begin(), first->sampleIdx.end(), 0);
}

void Nested::updateEllipseMean(ellipse_t &ellipse) const
{
  std::fill(ellipse.mean.begin(), ellipse.mean.end(), 0.);

  for (size_t i = 0; i < ellipse.num; ++i)
    for (size_t d = 0; d < _variableCount; ++d)
    {
      size_t sidx = ellipse.sampleIdx[i];
      ellipse.mean[d] += _liveSamples[sidx][d];
    }

  if (ellipse.num > 0)
    for (size_t d = 0; d < _variableCount; ++d)
      ellipse.mean[d] /= ((double)ellipse.num);
}

bool Nested::updateEllipseCov(ellipse_t &ellipse) const
{
  double weight = 1.0 / (ellipse.num - 1.0);

  if (ellipse.num <= ellipse.dim)
  {
    // update variance
    std::fill(ellipse.cov.begin(), ellipse.cov.end(), 0.);
    std::fill(ellipse.invCov.begin(), ellipse.invCov.end(), 0.);
    for (size_t d = 0; d < _variableCount; ++d)
    {
      double c = 0.0;
      for (size_t k = 0; k < ellipse.num; ++k)
      {
        size_t sidx = ellipse.sampleIdx[k];
        c += (_liveSamples[sidx][d] - ellipse.mean[d]) * (_liveSamples[sidx][d] - ellipse.mean[d]);
        ellipse.cov[d * _variableCount + d] = (ellipse.num == 1) ? 1.0 : weight * c;
        ellipse.invCov[d * _variableCount + d] = (ellipse.num == 1) ? 1.0 : 1. / (weight * c);
      }
    }
  }
  else
  {
    // update covariance
    for (size_t i = 0; i < _variableCount; ++i)
    {
      for (size_t j = i; j < _variableCount; ++j)
      {
        double c = 0.0;
        for (size_t k = 0; k < ellipse.num; ++k)
        {
          size_t sidx = ellipse.sampleIdx[k];
          c += (_liveSamples[sidx][i] - ellipse.mean[i]) * (_liveSamples[sidx][j] - ellipse.mean[j]);
        }
        ellipse.cov[j * _variableCount + i] = ellipse.cov[i * _variableCount + j] = weight * c;
      }
    }

    // update inverse covariance
    gsl_matrix_view cov = gsl_matrix_view_array(ellipse.cov.data(), _variableCount, _variableCount);

    gsl_matrix *matLU = gsl_matrix_alloc(_variableCount, _variableCount);
    gsl_matrix_memcpy(matLU, &cov.matrix);

    int signal;
    gsl_permutation *perm = gsl_permutation_alloc(_variableCount);
    gsl_linalg_LU_decomp(matLU, perm, &signal);

    std::fill(ellipse.invCov.begin(), ellipse.invCov.end(), 0.);
    gsl_matrix_view invCov = gsl_matrix_view_array(ellipse.invCov.data(), _variableCount, _variableCount);

    int status = gsl_linalg_LU_invert(matLU, perm, &invCov.matrix);
    gsl_permutation_free(perm);
    gsl_matrix_free(matLU);

    if (status != 0)
    {
      _k->_logger->logWarning("Normal", "Covariance inversion failed during ellipsoid covariance update.\n");
      return false;
    }
  }

  return true;
}

bool Nested::updateEllipseVolume(ellipse_t &ellipse) const
{
  if (ellipse.num == 0) return false;

  gsl_matrix_view cov = gsl_matrix_view_array(ellipse.cov.data(), _variableCount, _variableCount);

  gsl_matrix *matLU = gsl_matrix_alloc(_variableCount, _variableCount);
  int status = gsl_matrix_memcpy(matLU, &cov.matrix);
  if (status != 0)
  {
    _k->_logger->logWarning("Normal", "Memcpy failed ruing Ellipsoid volume update.\n");
    gsl_matrix_free(matLU);
    return false;
  }

  int signal;
  gsl_permutation *perm = gsl_permutation_alloc(_variableCount);
  gsl_linalg_LU_decomp(matLU, perm, &signal);
  gsl_permutation_free(perm);

  ellipse.det = gsl_linalg_LU_det(matLU, signal);
  gsl_matrix_free(matLU);

  gsl_vector_view evals = gsl_vector_view_array(ellipse.evals.data(), _variableCount);
  gsl_matrix_view paxes = gsl_matrix_view_array(ellipse.paxes.data(), _variableCount, _variableCount);

  gsl_matrix *matEigen = gsl_matrix_alloc(_variableCount, _variableCount);
  status = gsl_matrix_memcpy(matEigen, &cov.matrix);
  if (status != 0)
  {
    _k->_logger->logWarning("Normal", "Memcpy failed ruing Ellipsoid volume update.\n");
    gsl_matrix_free(matEigen);
    return false;
  }

  gsl_eigen_symmv_workspace *workEigen = gsl_eigen_symmv_alloc(_variableCount);
  status = gsl_eigen_symmv(matEigen, &evals.vector, &paxes.matrix, workEigen);
  gsl_matrix_free(matEigen);
  gsl_eigen_symmv_free(workEigen);
  if (status != 0)
  {
    _k->_logger->logWarning("Normal", "Eigenvalue Decomposition failed ruing Ellipsoid volume update.\n");
    return false;
  }

  status = gsl_eigen_symmv_sort(&evals.vector, &paxes.matrix, GSL_EIGEN_SORT_ABS_DESC);
  if (status != 0)
  {
    _k->_logger->logWarning("Normal", "Eigenvalue sorting failed ruing Ellipsoid volume update.\n");
    return false;
  }

  // calculate axes from cholesky decomposition
  gsl_matrix_view axes = gsl_matrix_view_array(ellipse.axes.data(), _variableCount, _variableCount);

  /* On output the diagonal and lower triangular part of the
     * input matrix A contain the matrix L, while the upper triangular part
     * contains the original matrix. */

  status = gsl_matrix_memcpy(&axes.matrix, &cov.matrix);
  if (status != 0)
  {
    _k->_logger->logWarning("Normal", "Memcpy failed ruing Ellipsoid volume update.\n");
    return false;
  }
  status = gsl_linalg_cholesky_decomp1(&axes.matrix); // LL^T = A
  if (status != 0)
  {
    _k->_logger->logWarning("Normal", "Cholesky Decomposition failed ruing Ellipsoid volume update.\n");
    return false;
  }

  // find scaling s.t. all samples are bounded by ellipse
  double res, max = std::numeric_limits<double>::lowest();
  for (size_t i = 0; i < ellipse.num; ++i)
  {
    size_t six = ellipse.sampleIdx[i];
    res = mahalanobisDistance(_liveSamples[six], ellipse);
    if (res > max) max = res;
  }

  ellipse.pointVolume = exp(_logVolume) * (double)ellipse.num / ((double)_numberLivePoints);

  double K = sqrt(pow(M_PI, _variableCount)) * 2.0 / ((double)_variableCount * gsl_sf_gamma(0.5 * _variableCount));
  double vol = sqrt(pow(_ellipsoidalScaling * max, _variableCount) * ellipse.det) * K;

  double enlargementFactor = vol > ellipse.pointVolume ? _ellipsoidalScaling * max : pow((ellipse.pointVolume * ellipse.pointVolume) / (K * K * ellipse.det), 1.0 / ((double)_variableCount));
  ellipse.volume = pow(enlargementFactor, _variableCount / 2.0) * sqrt(ellipse.det) * K;

  gsl_matrix_view invCov = gsl_matrix_view_array(ellipse.invCov.data(), _variableCount, _variableCount);

  // resize volume
  gsl_matrix_scale(&cov.matrix, enlargementFactor);
  gsl_matrix_scale(&invCov.matrix, 1. / enlargementFactor);
  gsl_vector_scale(&evals.vector, enlargementFactor);
  gsl_matrix_scale(&axes.matrix, sqrt(enlargementFactor));

  return true; //all good
}

double Nested::mahalanobisDistance(const std::vector<double> &sample, const ellipse_t &ellipse) const
{
  std::vector<double> dif(_variableCount);
  for (size_t d = 0; d < _variableCount; ++d) dif[d] = sample[d] - ellipse.mean[d];

  double tmp;
  double dist = 0.;
  for (size_t i = 0; i < _variableCount; ++i)
  {
    tmp = 0.0;
    for (size_t j = 0; j < _variableCount; ++j)
      tmp += dif[j] * ellipse.invCov[i + _variableCount * j];
    tmp *= dif[i];
    dist += tmp;
  }
  return dist;
}

double Nested::weightedMahalanobisDistance(const std::vector<double> &sample, const ellipse_t &ellipse) const
{
  double dist = mahalanobisDistance(sample, ellipse);

  return ellipse.volume * dist / ellipse.pointVolume;
}

bool Nested::kmeansClustering(const ellipse_t &parent, size_t maxIter, ellipse_t &childOne, ellipse_t &childTwo) const
{
  childOne.initSphere();
  childTwo.initSphere();

#ifdef WMDIST
  childOne.volume = 1.0;
  childOne.pointVolume = 1.0;
  childTwo.volume = 1.0;
  childTwo.pointVolume = 1.0;
#endif

  size_t ax = 0;
  for (size_t d = 0; d < _variableCount; ++d)
  {
    childOne.mean[d] = parent.mean[d] + parent.paxes[d * _variableCount + ax] * parent.evals[ax];
    childTwo.mean[d] = parent.mean[d] - parent.paxes[d * _variableCount + ax] * parent.evals[ax];
  }

  size_t nOne, nTwo, idxOne, idxTwo;
  std::vector<int8_t> clusterFlag(parent.num, 0);

  bool ok = true;
  size_t iter = 0;
  size_t diffs = 1;
  while ((diffs > 0) && (iter++ < maxIter))
  {
    diffs = 0;
    nOne = 0;
    nTwo = 0;

    // assign samples to means
    for (size_t i = 0; i < parent.num; ++i)
    {
      size_t six = parent.sampleIdx[i];

#ifdef L2DIST
      double d1 = l2distance(_liveSamples[six], childOne.mean);
      double d2 = l2distance(_liveSamples[six], childTwo.mean);
#else
  #ifdef WMDIST
      double d1 = weightedMahalanobisDistance(_liveSamples[six], childOne);
      double d2 = weightedMahalanobisDistance(_liveSamples[six], childTwo);
  #else
      double d1 = mahalanobisDistance(_liveSamples[six], childOne);
      double d2 = mahalanobisDistance(_liveSamples[six], childTwo);
  #endif
#endif
      int8_t flag = (d1 < d2) ? 1 : 2;

      if (clusterFlag[i] != flag) diffs++;
      clusterFlag[i] = flag;

      if (flag == 1)
        nOne++;
      else
        nTwo++;
    }

    childOne.num = nOne;
    childOne.sampleIdx.resize(nOne);

    childTwo.num = nTwo;
    childTwo.sampleIdx.resize(nTwo);

    idxOne = 0;
    idxTwo = 0;

    for (size_t i = 0; i < parent.num; ++i)
    {
      if (clusterFlag[i] == 1)
        childOne.sampleIdx[idxOne++] = parent.sampleIdx[i];
      else /* clusterFlag[i] == 2 */
        childTwo.sampleIdx[idxTwo++] = parent.sampleIdx[i];
    }

    updateEllipseMean(childOne);
    updateEllipseMean(childTwo);
#ifndef L2DIST
    ok = ok & updateEllipseCov(childOne);
    ok = ok & updateEllipseCov(childTwo);

  #ifdef WMDIST
    ok = ok & updateEllipseVolume(childOne);
    ok = ok & updateEllipseVolume(childTwo);
  #endif
#endif

    if (ok == false) break;
  }
  if (iter >= maxIter) _k->_logger->logWarning("Normal", "K-Means Clustering did not terminate in %zu steps.\n", maxIter);

  return ok;
}

void Nested::updateEffectiveSamples()
{
  double w = _sampleLogWeightDatabase.back();
  _sumLogWeights = safeLogPlus(_sumLogWeights, w);
  _sumSquareLogWeights = safeLogPlus(_sumSquareLogWeights, 2.0 * w);
  _effectiveSampleSize = exp(2.0 * _sumLogWeights - _sumSquareLogWeights);
}

bool Nested::insideUnitCube(const std::vector<double> &sample) const
{
  for (auto &s : sample)
  {
    if (s < 0.0) return false;
    if (s > 1.0) return false;
  }
  return true;
}

double Nested::safeLogPlus(double logx, double logy) const
{
  if (logx > logy)
    return logx + log1p(exp(logy - logx));
  else
    return logy + log1p(exp(logx - logy));
}

double Nested::safeLogMinus(double logx, double logy) const
{
  return log(exp(logx - logy) - 1) + logy;
}

void Nested::printGenerationBefore() { return; }

void Nested::printGenerationAfter()
{
  _k->_logger->logInfo("Minimal", "Log Evidence: %.4f (+- %.4f)\n", _logEvidence, sqrt(_logEvidenceVar));
  _k->_logger->logInfo("Minimal", "Sampling Efficiency: %.2f%%\n", 100.0 * _acceptedSamples / ((double)(_generatedSamples - _numberLivePoints)));
  _k->_logger->logInfo("Detailed", "Last Accepted: %zu\n", _lastAccepted);
  _k->_logger->logInfo("Detailed", "Effective Sample Size: %.2f\n", _effectiveSampleSize);
  _k->_logger->logInfo("Detailed", "Log Volume (shrinkage): %.2f/%.2f (%.2f%%)\n", _logVolume, _boundLogVolume, 100. * (1. - exp(_logVolume)));
  _k->_logger->logInfo("Normal", "lStar: %.2f (max llk evaluation %.2f)\n", _lStar, _maxEvaluation);
  _k->_logger->logInfo("Minimal", "Remaining Log Evidence: %.2f (dlogz: %.3f)\n", _remainingLogEvidence, _logEvidenceDifference);
  if (_resamplingMethod == "Multi Ellipse")
  {
    _k->_logger->logInfo("Detailed", "Num Ellipsoids: %zu\n", _ellipseVector.size());
  }
  return;
}

void Nested::finalize()
{
  if (_k->_currentGeneration <= 1) return;
  if (_addLivePoints == true) consumeLiveSamples();

  generatePosterior();

  _k->_logger->logInfo("Minimal", "Final Log Evidence: %.4f (+- %.4F)\n", _logEvidence, sqrt(_logEvidenceVar));
  _k->_logger->logInfo("Minimal", "Max evaluation: %.2f\n", _maxEvaluation);
  _k->_logger->logInfo("Minimal", "Sampling Efficiency: %.2f%%\n", 100.0 * _acceptedSamples / ((double)(_generatedSamples - _numberLivePoints)));
  return;
}

void Nested::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Normal Generator"))
 {
 _normalGenerator = dynamic_cast<korali::distribution::univariate::Normal*>(korali::Module::getModule(js["Normal Generator"], _k));
 _normalGenerator->applyVariableDefaults();
 _normalGenerator->applyModuleDefaults(js["Normal Generator"]);
 _normalGenerator->setConfiguration(js["Normal Generator"]);
   eraseValue(js, "Normal Generator");
 }

 if (isDefined(js, "Multivariate Generator"))
 {
 _multivariateGenerator = dynamic_cast<korali::distribution::multivariate::Normal*>(korali::Module::getModule(js["Multivariate Generator"], _k));
 _multivariateGenerator->applyVariableDefaults();
 _multivariateGenerator->applyModuleDefaults(js["Multivariate Generator"]);
 _multivariateGenerator->setConfiguration(js["Multivariate Generator"]);
   eraseValue(js, "Multivariate Generator");
 }

 if (isDefined(js, "Accepted Samples"))
 {
 try { _acceptedSamples = js["Accepted Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Accepted Samples']\n%s", e.what()); } 
   eraseValue(js, "Accepted Samples");
 }

 if (isDefined(js, "Generated Samples"))
 {
 try { _generatedSamples = js["Generated Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Generated Samples']\n%s", e.what()); } 
   eraseValue(js, "Generated Samples");
 }

 if (isDefined(js, "LogEvidence"))
 {
 try { _logEvidence = js["LogEvidence"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['LogEvidence']\n%s", e.what()); } 
   eraseValue(js, "LogEvidence");
 }

 if (isDefined(js, "LogEvidence Var"))
 {
 try { _logEvidenceVar = js["LogEvidence Var"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['LogEvidence Var']\n%s", e.what()); } 
   eraseValue(js, "LogEvidence Var");
 }

 if (isDefined(js, "LogVolume"))
 {
 try { _logVolume = js["LogVolume"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['LogVolume']\n%s", e.what()); } 
   eraseValue(js, "LogVolume");
 }

 if (isDefined(js, "Bound LogVolume"))
 {
 try { _boundLogVolume = js["Bound LogVolume"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Bound LogVolume']\n%s", e.what()); } 
   eraseValue(js, "Bound LogVolume");
 }

 if (isDefined(js, "Last Accepted"))
 {
 try { _lastAccepted = js["Last Accepted"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Last Accepted']\n%s", e.what()); } 
   eraseValue(js, "Last Accepted");
 }

 if (isDefined(js, "Next Update"))
 {
 try { _nextUpdate = js["Next Update"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Next Update']\n%s", e.what()); } 
   eraseValue(js, "Next Update");
 }

 if (isDefined(js, "Information"))
 {
 try { _information = js["Information"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Information']\n%s", e.what()); } 
   eraseValue(js, "Information");
 }

 if (isDefined(js, "LStar"))
 {
 try { _lStar = js["LStar"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['LStar']\n%s", e.what()); } 
   eraseValue(js, "LStar");
 }

 if (isDefined(js, "LStarOld"))
 {
 try { _lStarOld = js["LStarOld"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['LStarOld']\n%s", e.what()); } 
   eraseValue(js, "LStarOld");
 }

 if (isDefined(js, "LogWeight"))
 {
 try { _logWeight = js["LogWeight"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['LogWeight']\n%s", e.what()); } 
   eraseValue(js, "LogWeight");
 }

 if (isDefined(js, "Expected LogShrinkage"))
 {
 try { _expectedLogShrinkage = js["Expected LogShrinkage"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Expected LogShrinkage']\n%s", e.what()); } 
   eraseValue(js, "Expected LogShrinkage");
 }

 if (isDefined(js, "Max Evaluation"))
 {
 try { _maxEvaluation = js["Max Evaluation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Max Evaluation']\n%s", e.what()); } 
   eraseValue(js, "Max Evaluation");
 }

 if (isDefined(js, "Remaining Log Evidence"))
 {
 try { _remainingLogEvidence = js["Remaining Log Evidence"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Remaining Log Evidence']\n%s", e.what()); } 
   eraseValue(js, "Remaining Log Evidence");
 }

 if (isDefined(js, "Log Evidence Difference"))
 {
 try { _logEvidenceDifference = js["Log Evidence Difference"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Log Evidence Difference']\n%s", e.what()); } 
   eraseValue(js, "Log Evidence Difference");
 }

 if (isDefined(js, "Effective Sample Size"))
 {
 try { _effectiveSampleSize = js["Effective Sample Size"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Effective Sample Size']\n%s", e.what()); } 
   eraseValue(js, "Effective Sample Size");
 }

 if (isDefined(js, "Sum Log Weights"))
 {
 try { _sumLogWeights = js["Sum Log Weights"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Sum Log Weights']\n%s", e.what()); } 
   eraseValue(js, "Sum Log Weights");
 }

 if (isDefined(js, "Sum Square Log Weights"))
 {
 try { _sumSquareLogWeights = js["Sum Square Log Weights"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Sum Square Log Weights']\n%s", e.what()); } 
   eraseValue(js, "Sum Square Log Weights");
 }

 if (isDefined(js, "Prior Lower Bound"))
 {
 try { _priorLowerBound = js["Prior Lower Bound"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Prior Lower Bound']\n%s", e.what()); } 
   eraseValue(js, "Prior Lower Bound");
 }

 if (isDefined(js, "Prior Width"))
 {
 try { _priorWidth = js["Prior Width"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Prior Width']\n%s", e.what()); } 
   eraseValue(js, "Prior Width");
 }

 if (isDefined(js, "Candidates"))
 {
 try { _candidates = js["Candidates"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Candidates']\n%s", e.what()); } 
   eraseValue(js, "Candidates");
 }

 if (isDefined(js, "Candidate LogLikelihoods"))
 {
 try { _candidateLogLikelihoods = js["Candidate LogLikelihoods"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Candidate LogLikelihoods']\n%s", e.what()); } 
   eraseValue(js, "Candidate LogLikelihoods");
 }

 if (isDefined(js, "Candidate LogPriors"))
 {
 try { _candidateLogPriors = js["Candidate LogPriors"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Candidate LogPriors']\n%s", e.what()); } 
   eraseValue(js, "Candidate LogPriors");
 }

 if (isDefined(js, "Live Samples"))
 {
 try { _liveSamples = js["Live Samples"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Live Samples']\n%s", e.what()); } 
   eraseValue(js, "Live Samples");
 }

 if (isDefined(js, "Live LogLikelihoods"))
 {
 try { _liveLogLikelihoods = js["Live LogLikelihoods"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Live LogLikelihoods']\n%s", e.what()); } 
   eraseValue(js, "Live LogLikelihoods");
 }

 if (isDefined(js, "Live LogPriors"))
 {
 try { _liveLogPriors = js["Live LogPriors"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Live LogPriors']\n%s", e.what()); } 
   eraseValue(js, "Live LogPriors");
 }

 if (isDefined(js, "Live Samples Rank"))
 {
 try { _liveSamplesRank = js["Live Samples Rank"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Live Samples Rank']\n%s", e.what()); } 
   eraseValue(js, "Live Samples Rank");
 }

 if (isDefined(js, "Database Entries"))
 {
 try { _databaseEntries = js["Database Entries"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Database Entries']\n%s", e.what()); } 
   eraseValue(js, "Database Entries");
 }

 if (isDefined(js, "Sample Database"))
 {
 try { _sampleDatabase = js["Sample Database"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Sample Database']\n%s", e.what()); } 
   eraseValue(js, "Sample Database");
 }

 if (isDefined(js, "Sample LogLikelihood Database"))
 {
 try { _sampleLogLikelihoodDatabase = js["Sample LogLikelihood Database"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Sample LogLikelihood Database']\n%s", e.what()); } 
   eraseValue(js, "Sample LogLikelihood Database");
 }

 if (isDefined(js, "Sample LogPrior Database"))
 {
 try { _sampleLogPriorDatabase = js["Sample LogPrior Database"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Sample LogPrior Database']\n%s", e.what()); } 
   eraseValue(js, "Sample LogPrior Database");
 }

 if (isDefined(js, "Sample LogWeight Database"))
 {
 try { _sampleLogWeightDatabase = js["Sample LogWeight Database"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Sample LogWeight Database']\n%s", e.what()); } 
   eraseValue(js, "Sample LogWeight Database");
 }

 if (isDefined(js, "Covariance Matrix"))
 {
 try { _covarianceMatrix = js["Covariance Matrix"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Covariance Matrix']\n%s", e.what()); } 
   eraseValue(js, "Covariance Matrix");
 }

 if (isDefined(js, "Log Domain Size"))
 {
 try { _logDomainSize = js["Log Domain Size"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Log Domain Size']\n%s", e.what()); } 
   eraseValue(js, "Log Domain Size");
 }

 if (isDefined(js, "Domain Mean"))
 {
 try { _domainMean = js["Domain Mean"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Domain Mean']\n%s", e.what()); } 
   eraseValue(js, "Domain Mean");
 }

 if (isDefined(js, "Box Lower Bound"))
 {
 try { _boxLowerBound = js["Box Lower Bound"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Box Lower Bound']\n%s", e.what()); } 
   eraseValue(js, "Box Lower Bound");
 }

 if (isDefined(js, "Box Upper Bound"))
 {
 try { _boxUpperBound = js["Box Upper Bound"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Box Upper Bound']\n%s", e.what()); } 
   eraseValue(js, "Box Upper Bound");
 }

 if (isDefined(js, "Ellipse Axes"))
 {
 try { _ellipseAxes = js["Ellipse Axes"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Ellipse Axes']\n%s", e.what()); } 
   eraseValue(js, "Ellipse Axes");
 }

 if (isDefined(js, "Number Live Points"))
 {
 try { _numberLivePoints = js["Number Live Points"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Number Live Points']\n%s", e.what()); } 
   eraseValue(js, "Number Live Points");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Number Live Points'] required by Nested.\n"); 

 if (isDefined(js, "Batch Size"))
 {
 try { _batchSize = js["Batch Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Batch Size']\n%s", e.what()); } 
   eraseValue(js, "Batch Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Batch Size'] required by Nested.\n"); 

 if (isDefined(js, "Add Live Points"))
 {
 try { _addLivePoints = js["Add Live Points"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Add Live Points']\n%s", e.what()); } 
   eraseValue(js, "Add Live Points");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Add Live Points'] required by Nested.\n"); 

 if (isDefined(js, "Resampling Method"))
 {
 try { _resamplingMethod = js["Resampling Method"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Resampling Method']\n%s", e.what()); } 
   eraseValue(js, "Resampling Method");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Resampling Method'] required by Nested.\n"); 

 if (isDefined(js, "Proposal Update Frequency"))
 {
 try { _proposalUpdateFrequency = js["Proposal Update Frequency"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Proposal Update Frequency']\n%s", e.what()); } 
   eraseValue(js, "Proposal Update Frequency");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Proposal Update Frequency'] required by Nested.\n"); 

 if (isDefined(js, "Ellipsoidal Scaling"))
 {
 try { _ellipsoidalScaling = js["Ellipsoidal Scaling"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Ellipsoidal Scaling']\n%s", e.what()); } 
   eraseValue(js, "Ellipsoidal Scaling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Ellipsoidal Scaling'] required by Nested.\n"); 

 if (isDefined(js, "Termination Criteria", "Min Log Evidence Delta"))
 {
 try { _minLogEvidenceDelta = js["Termination Criteria"]["Min Log Evidence Delta"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Termination Criteria']['Min Log Evidence Delta']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Min Log Evidence Delta");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Min Log Evidence Delta'] required by Nested.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Effective Sample Size"))
 {
 try { _maxEffectiveSampleSize = js["Termination Criteria"]["Max Effective Sample Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Termination Criteria']['Max Effective Sample Size']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Effective Sample Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Effective Sample Size'] required by Nested.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Log Likelihood"))
 {
 try { _maxLogLikelihood = js["Termination Criteria"]["Max Log Likelihood"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Termination Criteria']['Max Log Likelihood']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Log Likelihood");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Log Likelihood'] required by Nested.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Sampler::setConfiguration(js);
 _type = "sampler/Nested";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: Nested: \n%s\n", js.dump(2).c_str());
} 

void Nested::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Number Live Points"] = _numberLivePoints;
   js["Batch Size"] = _batchSize;
   js["Add Live Points"] = _addLivePoints;
   js["Resampling Method"] = _resamplingMethod;
   js["Proposal Update Frequency"] = _proposalUpdateFrequency;
   js["Ellipsoidal Scaling"] = _ellipsoidalScaling;
   js["Termination Criteria"]["Min Log Evidence Delta"] = _minLogEvidenceDelta;
   js["Termination Criteria"]["Max Effective Sample Size"] = _maxEffectiveSampleSize;
   js["Termination Criteria"]["Max Log Likelihood"] = _maxLogLikelihood;
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
 if(_multivariateGenerator != NULL) _multivariateGenerator->getConfiguration(js["Multivariate Generator"]);
   js["Accepted Samples"] = _acceptedSamples;
   js["Generated Samples"] = _generatedSamples;
   js["LogEvidence"] = _logEvidence;
   js["LogEvidence Var"] = _logEvidenceVar;
   js["LogVolume"] = _logVolume;
   js["Bound LogVolume"] = _boundLogVolume;
   js["Last Accepted"] = _lastAccepted;
   js["Next Update"] = _nextUpdate;
   js["Information"] = _information;
   js["LStar"] = _lStar;
   js["LStarOld"] = _lStarOld;
   js["LogWeight"] = _logWeight;
   js["Expected LogShrinkage"] = _expectedLogShrinkage;
   js["Max Evaluation"] = _maxEvaluation;
   js["Remaining Log Evidence"] = _remainingLogEvidence;
   js["Log Evidence Difference"] = _logEvidenceDifference;
   js["Effective Sample Size"] = _effectiveSampleSize;
   js["Sum Log Weights"] = _sumLogWeights;
   js["Sum Square Log Weights"] = _sumSquareLogWeights;
   js["Prior Lower Bound"] = _priorLowerBound;
   js["Prior Width"] = _priorWidth;
   js["Candidates"] = _candidates;
   js["Candidate LogLikelihoods"] = _candidateLogLikelihoods;
   js["Candidate LogPriors"] = _candidateLogPriors;
   js["Live Samples"] = _liveSamples;
   js["Live LogLikelihoods"] = _liveLogLikelihoods;
   js["Live LogPriors"] = _liveLogPriors;
   js["Live Samples Rank"] = _liveSamplesRank;
   js["Database Entries"] = _databaseEntries;
   js["Sample Database"] = _sampleDatabase;
   js["Sample LogLikelihood Database"] = _sampleLogLikelihoodDatabase;
   js["Sample LogPrior Database"] = _sampleLogPriorDatabase;
   js["Sample LogWeight Database"] = _sampleLogWeightDatabase;
   js["Covariance Matrix"] = _covarianceMatrix;
   js["Log Domain Size"] = _logDomainSize;
   js["Domain Mean"] = _domainMean;
   js["Box Lower Bound"] = _boxLowerBound;
   js["Box Upper Bound"] = _boxUpperBound;
   js["Ellipse Axes"] = _ellipseAxes;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Sampler::getConfiguration(js);
} 

void Nested::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Number Live Points\": 1500, \"Batch Size\": 1, \"Add Live Points\": true, \"Resampling Method\": \"Ellipse\", \"Proposal Update Frequency\": 1500, \"Ellipsoidal Scaling\": 1.0, \"Termination Criteria\": {\"Min Log Evidence Delta\": 0.01, \"Max Effective Sample Size\": 10000000.0, \"Max Log Likelihood\": 10000000.0}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}, \"Normal Generator\": {\"Type\": \"Univariate/Normal\", \"Mean\": 0.0, \"Standard Deviation\": 1.0}, \"Multivariate Generator\": {\"Type\": \"Multivariate/Normal\"}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Sampler::applyModuleDefaults(js);
} 

void Nested::applyVariableDefaults() 
{

 Sampler::applyVariableDefaults();
} 

bool Nested::checkTermination()
{
 bool hasFinished = false;

 if ((_k->_currentGeneration > 1) && (_logEvidenceDifference <= _minLogEvidenceDelta))
 {
  _terminationCriteria.push_back("Nested['Min Log Evidence Delta'] = " + std::to_string(_minLogEvidenceDelta) + ".");
  hasFinished = true;
 }

 if (_maxEffectiveSampleSize <= _effectiveSampleSize)
 {
  _terminationCriteria.push_back("Nested['Max Effective Sample Size'] = " + std::to_string(_maxEffectiveSampleSize) + ".");
  hasFinished = true;
 }

 if (_maxLogLikelihood <= _lStar)
 {
  _terminationCriteria.push_back("Nested['Max Log Likelihood'] = " + std::to_string(_maxLogLikelihood) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Sampler::checkTermination();
 return hasFinished;
}



} //sampler
} //solver
} //korali

