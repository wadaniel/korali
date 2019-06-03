#include "korali.h"
#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <numeric>   // std::iota
#include <algorithm> // std::sort
using namespace Korali::Solver;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

CMAES::CMAES(nlohmann::json& js) : Korali::Solver::Base::Base(js)
{
 setConfiguration(js);

 // Allocating Memory
 samplePopulation  = (double*) calloc (sizeof(double), _k->_problem->N*_s);
 rgpc              = (double*) calloc (sizeof(double), _k->_problem->N);
 rgps              = (double*) calloc (sizeof(double), _k->_problem->N);
 rgdTmp            = (double*) calloc (sizeof(double), _k->_problem->N);
 rgBDz             = (double*) calloc (sizeof(double), _k->_problem->N);
 rgxmean           = (double*) calloc (sizeof(double), _k->_problem->N);
 rgxold            = (double*) calloc (sizeof(double), _k->_problem->N);
 rgxbestever       = (double*) calloc (sizeof(double), _k->_problem->N);
 rgD               = (double*) calloc (sizeof(double), _k->_problem->N);
 curBestVector     = (double*) calloc (sizeof(double), _k->_problem->N);

 histFuncValues = (double*) calloc (sizeof(double), _termCondMaxGenerations+1);
 index          = (size_t*) calloc (sizeof(size_t), _s);

 C = (double**) calloc (sizeof(double*), _k->_problem->N);
 B = (double**) calloc (sizeof(double*), _k->_problem->N);
 for (size_t i = 0; i < _k->_problem->N; i++) C[i] = (double*) calloc (sizeof(double), _k->_problem->N);
 for (size_t i = 0; i < _k->_problem->N; i++) B[i] = (double*) calloc (sizeof(double), _k->_problem->N);

 initializedSample = (bool*) calloc (sizeof(bool), _s);
 fitnessVector = (double*) calloc (sizeof(double), _s);

 // Init Generation
 _currentGeneration = 0;
 
 // Initializing Gaussian Generator
 _gaussianGenerator = new Variable::Gaussian(0.0, 1.0, _k->_seed++);

 _chiN = sqrt((double) _k->_problem->N) * (1. - 1./(4.*_k->_problem->N) + 1./(21.*_k->_problem->N*_k->_problem->N));
 
 // Setting Sigma Cumulation Factor
 if (_sigmaCumulationFactor <= 0 || _sigmaCumulationFactor > 1) _sigmaCumulationFactor = (_muEffective + 2.) / (_k->_problem->N + _muEffective + 3.0);

 // Setting Damping Factor

 if (_dampFactor <= 0)
 { 
   _dampFactor = 1.0* (1 + 2*std::max(0.0, sqrt((_muEffective-1.0)/(_k->_problem->N+1.0)) - 1))  /* basic factor */
     //* std::max(0.3, 1. - (double)_k->_problem->N / (1e-6+std::min(_termCondMaxGenerations, _termCondMaxFitnessEvaluations/_s))) /* modification for short runs */
     + _sigmaCumulationFactor; /* minor increment */
 }

 // Setting Cumulative Covariance

 if (_cumulativeCovariance <= 0 || _cumulativeCovariance> 1)  _cumulativeCovariance = 4. / (_k->_problem->N + 4);

 // Set covariance Matrix Learning Rate

 double l1 = 2. / ((_k->_problem->N+1.4142)*(_k->_problem->N+1.4142));
 double l2 = (2.*_muEffective-1.) / ((_k->_problem->N+2.)*(_k->_problem->N+2.)+_muEffective);
 l2 = (l2 > 1) ? 1 : l2;
 l2 = (1./_muCovariance) * l1 + (1.-1./_muCovariance) * l2;

 if (_covarianceMatrixLearningRate < 0 || _covarianceMatrixLearningRate > 1)  _covarianceMatrixLearningRate = l2;

 // Setting eigensystem evaluation Frequency

 if( _covarianceEigenEvalFreq < 0.0) _covarianceEigenEvalFreq = 1.0/_covarianceMatrixLearningRate/((double)_k->_problem->N)/10.0;

 double trace = 0.0;
 for (size_t i = 0; i < _k->_problem->N; ++i) trace += _k->_problem->_variables[i]->_initialStdDev*_k->_problem->_variables[i]->_initialStdDev;
 sigma = sqrt(trace/_k->_problem->N); /* _muEffective/(0.2*_muEffective+sqrt(_k->_problem->N)) * sqrt(trace/_k->_problem->N); */

 flgEigensysIsUptodate = true;

 countevals = 0;
 countinfeasible = 0;
 bestEver = 0.0;

 for (size_t i = 0; i < _k->_problem->N; ++i)
 {
  B[i][i] = 1.0;
  C[i][i] = rgD[i] = _k->_problem->_variables[i]->_initialStdDev * sqrt(_k->_problem->N / trace);
  C[i][i] *= C[i][i];
 }

 minEW = doubleRangeMin(rgD, _k->_problem->N); minEW = minEW * minEW;
 maxEW = doubleRangeMax(rgD, _k->_problem->N); maxEW = maxEW * maxEW;

 maxdiagC=C[0][0]; for(size_t i=1;i<_k->_problem->N;++i) if(maxdiagC<C[i][i]) maxdiagC=C[i][i];
 mindiagC=C[0][0]; for(size_t i=1;i<_k->_problem->N;++i) if(mindiagC>C[i][i]) mindiagC=C[i][i];

 psL2 = 0.0;

 /* set rgxmean */
 for (size_t i = 0; i < _k->_problem->N; ++i)
 {
   if(_k->_problem->_variables[i]->_initialValue < _k->_problem->_variables[i]->_lowerBound || _k->_problem->_variables[i]->_initialValue > _k->_problem->_variables[i]->_upperBound)
    fprintf(stderr,"[Korali] Warning: Initial Value (%.4f) for \'%s\' is out of bounds (%.4f-%.4f).\n", _k->_problem->_variables[i]->_initialValue, _k->_problem->_variables[i]->_name.c_str(), _k->_problem->_variables[i]->_lowerBound, _k->_problem->_variables[i]->_upperBound);
   rgxmean[i] = rgxold[i] = _k->_problem->_variables[i]->_initialValue;
 }

 // If state is defined:
 if (isDefined(js, {"State"}))
 {
  setState(js);
  js.erase("State");
 }
}

CMAES::~CMAES()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json CMAES::getConfiguration()
{
 auto js = this->Korali::Solver::Base::getConfiguration();

 js["Method"] = "CMA-ES";

 js["Lambda"]                  = _s;
 js["Sigma Cumulation Factor"] = _sigmaCumulationFactor;
 js["Damp Factor"]             = _dampFactor;
 js["Objective"]               = _objective;

 js["Mu"]["Value"]      = _mu;
 js["Mu"]["Type"]       = _muType;
 js["Mu"]["Covariance"] = _muCovariance;

 js["Covariance Matrix"]["Cumulative Covariance"]           = _cumulativeCovariance;
 js["Covariance Matrix"]["Learning Rate"]                   = _covarianceMatrixLearningRate;
 js["Covariance Matrix"]["Eigenvalue Evaluation Frequency"] = _covarianceEigenEvalFreq;
 js["Covariance Matrix"]["Is Diagonal"]                     = _isdiag;

 js["Termination Criteria"]["Max Generations"]["Value"]           = _termCondMaxGenerations;
 js["Termination Criteria"]["Max Generations"]["Active"]          = _isTermCondMaxGenerations;
 js["Termination Criteria"]["Max Model Evaluations"]["Value"]     = _termCondMaxFitnessEvaluations;
 js["Termination Criteria"]["Max Model Evaluations"]["Active"]    = _isTermCondMaxFitnessEvaluations;
 js["Termination Criteria"]["Min Fitness"]["Value"]               = _termCondFitness;
 js["Termination Criteria"]["Min Fitness"]["Active"]              = _isTermCondFitness;
 js["Termination Criteria"]["Fitness Diff Threshold"]["Value"]    = _termCondFitnessDiffThreshold;
 js["Termination Criteria"]["Fitness Diff Threshold"]["Active"]   = _isTermCondFitnessDiffThreshold;
 js["Termination Criteria"]["Min DeltaX"]["Value"]                = _termCondMinDeltaX;
 js["Termination Criteria"]["Min DeltaX"]["Active"]               = _isTermCondMinDeltaX;
 js["Termination Criteria"]["Max Standard Deviation"]["Value"]    = _termCondTolUpXFactor;
 js["Termination Criteria"]["Max Standard Deviation"]["Active"]   = _isTermCondTolUpXFactor;
 js["Termination Criteria"]["Max Condition Covariance"]["Value"]  = _termCondCovCond;
 js["Termination Criteria"]["Max Condition Covariance"]["Active"] = _isTermCondCovCond;

 // State Variables
 js["State"]["Current Generation"]        = _currentGeneration;
 js["State"]["Sigma"]                     = sigma;
 js["State"]["BestEverFunctionValue"]     = bestEver;
 js["State"]["PreviousBestFunctionValue"] = prevBest;
 js["State"]["CurrentBestFunctionValue"]  = currentFunctionValue;
 js["State"]["PrevFunctionValue"]         = prevFunctionValue;
 js["State"]["MaxDiagonalCovariance"]     = maxdiagC;
 js["State"]["MinDiagonalCovariance"]     = mindiagC;
 js["State"]["MaxEigenvalue"]             = maxEW;
 js["State"]["MinEigenvalue"]             = minEW;
 js["State"]["EigenSystemUpToDate"]       = flgEigensysIsUptodate;
 js["State"]["EvaluationCount"]           = countevals;
 js["State"]["InfeasibleCount"]           = countinfeasible;
 js["State"]["ConjugateEvolutionPathL2"]  = psL2;

 for (size_t i = 0; i < _s; i++) js["State"]["Index"]          += index[i];
 for (size_t i = 0; i < _s; i++) js["State"]["FunctionValues"] += fitnessVector[i];

 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["CurrentMeanVector"]      += rgxmean[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["PreviousMeanVector"]     += rgxold[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["BestEverVector"]         += rgxbestever[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["CurrentBestVector"]      += curBestVector[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["AxisLengths"]            += rgD[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["EvolutionPath"]          += rgpc[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["ConjugateEvolutionPath"] += rgps[i];

 for (size_t i = 0; i < _s; i++) for (size_t j = 0; j < _k->_problem->N; j++) js["State"]["Samples"][i][j] = samplePopulation[i*_k->_problem->N + j];
 for (size_t i = 0; i < _k->_problem->N; i++) for (size_t j = 0; j < _k->_problem->N; j++) js["State"]["CovarianceMatrix"][i][j] = C[i][j];
 for (size_t i = 0; i < _k->_problem->N; i++) for (size_t j = 0; j < _k->_problem->N; j++) js["State"]["EigenMatrix"][i][j] = B[i][j];

 return js;
}



void CMAES::setConfiguration(nlohmann::json& js)
{
 this->Korali::Solver::Base::setConfiguration(js);

 _s                     = consume(js, { "Lambda" }, KORALI_NUMBER);
 _sigmaCumulationFactor = consume(js, { "Sigma Cumulation Factor" }, KORALI_NUMBER, std::to_string(-1));
 _dampFactor            = consume(js, { "Damp Factor" }, KORALI_NUMBER, std::to_string(-1));
 _objective             = consume(js, { "Objective" }, KORALI_STRING, "Maximize");

 _fitnessSign = 0;
 if(_objective == "Maximize") _fitnessSign = 1;
 if(_objective == "Minimize") _fitnessSign = -1;
 if(_fitnessSign == 0)  { fprintf( stderr, "[Korali] Error: Invalid setting for Objective: %s\n", _objective.c_str()); exit(-1); }

 _mu            = consume(js, { "Mu", "Value" }, KORALI_NUMBER, std::to_string(ceil(_s / 2)));
 _muType        = consume(js, { "Mu", "Type" }, KORALI_STRING, "Logarithmic");
 _muCovariance  = consume(js, { "Mu", "Covariance" }, KORALI_NUMBER, std::to_string(-1));

 // Initializing Mu Weights
 _muWeights = (double *) calloc (sizeof(double), _mu);
 if (_muType == "Linear")       for (size_t i = 0; i < _mu; i++)  _muWeights[i] = _mu - i;
 else if (_muType == "Equal")        for (size_t i = 0; i < _mu; i++)  _muWeights[i] = 1;
 else if (_muType == "Logarithmic")  for (size_t i = 0; i < _mu; i++)  _muWeights[i] = log(_mu+1.)-log(i+1.);
 else  { fprintf( stderr, "[Korali] Error: Invalid setting of Mu Type (%s) (Linear, Equal or Logarithmic accepted).", _muType); exit(-1); }
 
 
 /* normalize weights vector and set mueff */
 double s1 = 0.0;
 double s2 = 0.0;

 for (size_t  i = 0; i < _mu; i++)
 {
  s1 += _muWeights[i];
  s2 += _muWeights[i]*_muWeights[i];
 }
 _muEffective = s1*s1/s2;

 for (size_t i = 0; i < _mu; i++) _muWeights[i] /= s1;

 if(_mu < 1 || _mu > _s || (_mu == _s && _muWeights[0] == _muWeights[_mu-1]))
 { fprintf( stderr, "[Korali] Error: Invalid setting of Mu (%lu) and/or Lambda (%lu)\n", _mu, _s); exit(-1); }
 // Setting MU Covariance
 if (_muCovariance < 1) _muCovariance = _muEffective;

 _covarianceEigenEvalFreq         = consume(js, { "Covariance Matrix", "Eigenvalue Evaluation Frequency" }, KORALI_NUMBER, std::to_string(-1));
 _cumulativeCovariance            = consume(js, { "Covariance Matrix", "Cumulative Covariance" }, KORALI_NUMBER, std::to_string(-1));
 _covarianceMatrixLearningRate    = consume(js, { "Covariance Matrix", "Learning Rate" }, KORALI_NUMBER, std::to_string(-1));
 _isdiag                          = consume(js, { "Covariance Matrix", "Is Diagonal" }, KORALI_BOOLEAN, "false");

 _termCondMaxGenerations          = consume(js, { "Termination Criteria", "Max Generations", "Value" }, KORALI_NUMBER, std::to_string(1000));
 _isTermCondMaxGenerations        = consume(js, { "Termination Criteria", "Max Generations", "Active" }, KORALI_BOOLEAN, "true");
 _termCondMaxFitnessEvaluations   = consume(js, { "Termination Criteria", "Max Model Evaluations", "Value" }, KORALI_NUMBER, std::to_string(std::numeric_limits<size_t>::max()));
 _isTermCondMaxFitnessEvaluations = consume(js, { "Termination Criteria", "Max Model Evaluations", "Active" }, KORALI_BOOLEAN, "true");
 _termCondFitness                 = consume(js, { "Termination Criteria", "Fitness", "Value" }, KORALI_NUMBER, std::to_string(std::numeric_limits<double>::max()));
 _isTermCondFitness               = consume(js, { "Termination Criteria", "Fitness", "Active" }, KORALI_BOOLEAN, "true");
 _termCondFitnessDiffThreshold    = consume(js, { "Termination Criteria", "Fitness Diff Threshold", "Value" }, KORALI_NUMBER, std::to_string(1e-9));
 _isTermCondFitnessDiffThreshold  = consume(js, { "Termination Criteria", "Fitness Diff Threshold", "Active" }, KORALI_BOOLEAN, "true");
 _termCondMinDeltaX               = consume(js, { "Termination Criteria", "Min DeltaX", "Value" }, KORALI_NUMBER, std::to_string(0.0));
 _isTermCondMinDeltaX             = consume(js, { "Termination Criteria", "Min DeltaX", "Active" }, KORALI_BOOLEAN, "true");
 _termCondTolUpXFactor            = consume(js, { "Termination Criteria", "Max Standard Deviation", "Value" }, KORALI_NUMBER, std::to_string(1e18));
 _isTermCondTolUpXFactor          = consume(js, { "Termination Criteria", "Max Standard Deviation", "Active" }, KORALI_BOOLEAN, "true");
 _termCondCovCond                 = consume(js, { "Termination Criteria", "Max Condition Covariance", "Value" }, KORALI_NUMBER, std::to_string(std::numeric_limits<double>::max()));
 _isTermCondCovCond               = consume(js, { "Termination Criteria", "Max Condition Covariance", "Active" }, KORALI_BOOLEAN, "true");

}



void CMAES::setState(nlohmann::json& js)
{
 this->Korali::Solver::Base::setState(js);

 _currentGeneration    = js["State"]["Current Generation"];
 sigma                 = js["State"]["Sigma"];
 bestEver              = js["State"]["BestEverFunctionValue"];
 prevBest              = js["State"]["PreviousBestFunctionValue"];
 currentFunctionValue  = js["State"]["CurrentBestFunctionValue"];
 prevFunctionValue     = js["State"]["PrevFunctionValue"];
 maxdiagC              = js["State"]["MaxDiagonalCovariance"];
 mindiagC              = js["State"]["MinDiagonalCovariance"];
 maxEW                 = js["State"]["MaxEigenvalue"];
 minEW                 = js["State"]["MinEigenvalue"] ;
 flgEigensysIsUptodate = js["State"]["EigenSystemUpToDate"];
 countevals            = js["State"]["EvaluationCount"];
 countinfeasible       = js["State"]["InfeasibleCount"];
 psL2                  = js["State"]["ConjugateEvolutionPathL2"];

 for (size_t i = 0; i < _s; i++) index[i]         = js["State"]["Index"][i];
 for (size_t i = 0; i < _s; i++) fitnessVector[i] = js["State"]["FunctionValues"][i];

 for (size_t i = 0; i < _k->_problem->N; i++) rgxmean[i]       = js["State"]["CurrentMeanVector"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) rgxold[i]        = js["State"]["PreviousMeanVector"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) rgxbestever[i]   = js["State"]["BestEverVector"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) curBestVector[i] = js["State"]["CurrentBestVector"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) rgD[i]           = js["State"]["AxisLengths"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) rgpc[i]          = js["State"]["EvolutionPath"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) rgps[i]          = js["State"]["ConjugateEvolutionPath"][i];
 for (size_t i = 0; i < _s; i++) for (size_t j = 0; j < _k->_problem->N; j++) samplePopulation[i*_k->_problem->N + j] = js["State"]["Samples"][i][j];
 for (size_t i = 0; i < _k->_problem->N; i++) for (size_t j = 0; j < _k->_problem->N; j++) C[i][j] = js["State"]["CovarianceMatrix"][i][j];
 for (size_t i = 0; i < _k->_problem->N; i++) for (size_t j = 0; j < _k->_problem->N; j++) B[i][j] = js["State"]["EigenMatrix"][i][j];
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void CMAES::run()
{
 if (_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Starting CMA-ES (Objective: %s).\n", _objective.c_str());

  startTime = std::chrono::system_clock::now();
  _k->saveState(_currentGeneration);

  while(!checkTermination())
  {
    t0 = std::chrono::system_clock::now();
    
    prepareGeneration();
    evaluateSamples();
    updateDistribution(fitnessVector);

    t1 = std::chrono::system_clock::now();
    printGeneration();

    _k->saveState(_currentGeneration);
  }

  endTime = std::chrono::system_clock::now();
  printFinal();

}


void CMAES::evaluateSamples() 
{
  while (finishedSamples < _s)
  {
    for (size_t i = 0; i < _s; i++) if (initializedSample[i] == false)
    {
      initializedSample[i] = true;
      _k->_conduit->evaluateSample(samplePopulation, i);
    }
    _k->_conduit->checkProgress();
  }
}


void CMAES::processSample(size_t sampleId, double fitness)
{
 fitnessVector[sampleId] = _fitnessSign*fitness;
 finishedSamples++;
}


bool CMAES::isFeasible(size_t sampleIdx) const
{
 for (size_t d = 0; d < _k->_problem->N; ++d)
  if (samplePopulation[ sampleIdx*_k->_problem->N+d ] < _k->_problem->_variables[d]->_lowerBound || samplePopulation[ sampleIdx*_k->_problem->N+d ] > _k->_problem->_variables[d]->_upperBound) return false;
 return true;
}


void CMAES::prepareGeneration()
{

 /* calculate eigensystem  */
 if (!flgEigensysIsUptodate) {
  if (!_isdiag)
   updateEigensystem(0);
  else {
   for (size_t i = 0; i < _k->_problem->N; ++i)
    rgD[i] = sqrt(C[i][i]);
   minEW = doubleRangeMin(rgD, _k->_problem->N) * doubleRangeMin(rgD, _k->_problem->N);
   maxEW = doubleRangeMax(rgD, _k->_problem->N) * doubleRangeMin(rgD, _k->_problem->N);
   flgEigensysIsUptodate = true;
  }
 }

 for (size_t iNk = 0; iNk < _s; ++iNk) 
 {
     sampleSingle(iNk);
     while( !isFeasible( iNk )) { countinfeasible++; sampleSingle(iNk); }
 }

 _currentGeneration++;

 finishedSamples = 0;
 for (size_t i = 0; i < _s; i++) initializedSample[i] = false;
}


void CMAES::sampleSingle(size_t sampleIdx) 
{
  
  /* generate scaled random vector (D * z) */
  for (size_t d = 0; d < _k->_problem->N; ++d)
  {
   if (_isdiag) samplePopulation[sampleIdx * _k->_problem->N + d] = rgxmean[d] + sigma * rgD[d] * _gaussianGenerator->getRandomNumber();
   else rgdTmp[d] = rgD[d] * _gaussianGenerator->getRandomNumber();
  }

  if (!_isdiag)
   for (size_t d = 0; d < _k->_problem->N; ++d) {
   double sum = 0.0;
   for (size_t e = 0; e < _k->_problem->N; ++e) sum += B[d][e] * rgdTmp[d];
    samplePopulation[sampleIdx * _k->_problem->N + d] = rgxmean[d] + sigma * sum;
  }
}


void CMAES::updateDistribution(const double *fitnessVector)
{
 countevals += _s;

 /* Generate index */
 sort_index(fitnessVector, index, _s);

 /* update function value history */
 prevFunctionValue = currentFunctionValue;

 /* update current best */
 currentFunctionValue = fitnessVector[index[0]];
 for (size_t i = 0; i < _k->_problem->N; ++i) curBestVector[i] = samplePopulation[index[0]*_k->_problem->N + i];

 /* update xbestever */
 if (currentFunctionValue > bestEver || _currentGeneration == 1)
 {
  prevBest = bestEver;
  bestEver = currentFunctionValue;
  for (size_t i = 0; i < _k->_problem->N; ++i) rgxbestever[i] = curBestVector[i];
 }

 histFuncValues[_currentGeneration] = bestEver;

 /* calculate rgxmean and rgBDz */
 for (size_t i = 0; i < _k->_problem->N; ++i) {
   rgxold[i] = rgxmean[i];
   rgxmean[i] = 0.;
   for (size_t iNk = 0; iNk < _mu; ++iNk)
     rgxmean[i] += _muWeights[iNk] * samplePopulation[index[iNk]*_k->_problem->N + i];

   rgBDz[i] = (rgxmean[i] - rgxold[i])/sigma;
 }

 /* calculate z := D^(-1) * B^(T) * rgBDz into rgdTmp */
 for (size_t i = 0; i < _k->_problem->N; ++i) {
  double sum = 0.0;
  if (_isdiag) sum = rgBDz[i];
  else for (size_t j = 0; j < _k->_problem->N; ++j) sum += B[j][i] * rgBDz[j]; /* B^(T) * rgBDz */

  rgdTmp[i] = sum / rgD[i]; /* D^(-1) */
 }

 psL2 = 0.0;

 /* cumulation for sigma (ps) using B*z */
 for (size_t i = 0; i < _k->_problem->N; ++i) {
    double sum = 0.0;
    if (_isdiag) sum = rgdTmp[i];
    else for (size_t j = 0; j < _k->_problem->N; ++j) sum += B[i][j] * rgdTmp[j];

    rgps[i] = (1. - _sigmaCumulationFactor) * rgps[i] + sqrt(_sigmaCumulationFactor * (2. - _sigmaCumulationFactor) * _muEffective) * sum;

    /* calculate norm(ps)^2 */
    psL2 += rgps[i] * rgps[i];
 }

 int hsig = sqrt(psL2) / sqrt(1. - pow(1.-_sigmaCumulationFactor, 2*_currentGeneration)) / _chiN  < 1.4 + 2./(_k->_problem->N+1);

 /* cumulation for covariance matrix (pc) using B*D*z~_k->_problem->N(0,C) */
 for (size_t i = 0; i < _k->_problem->N; ++i)
   rgpc[i] = (1. - _cumulativeCovariance) * rgpc[i] + hsig * sqrt( _cumulativeCovariance * (2. - _cumulativeCovariance) *_muEffective ) * rgBDz[i];

 /* update of C  */
 adaptC(hsig);

 /* update of sigma */
 sigma *= exp(((sqrt(psL2)/_chiN)-1.)*_sigmaCumulationFactor/_dampFactor);

 /* numerical error management */

 //treat maximal standnard deviations
 //TODO

 //treat minimal standard deviations
 for (size_t i = 0; i < _k->_problem->N; ++i)
   if (sigma * sqrt(C[i][i]) < _k->_problem->_variables[i]->_minStdDevChange) {
     sigma = (_k->_problem->_variables[i]->_minStdDevChange)/sqrt(C[i][i]) * exp(0.05+_sigmaCumulationFactor/_dampFactor);
     if (_k->_verbosity >= KORALI_DETAILED) fprintf(stderr, "[Korali] Warning: sigma increased due to minimal standard deviation.\n");
   }

 //too low coordinate axis deviations
 //TODO

 //treat numerical precision provblems
 //TODO

 /* Test if function values are identical, escape flat fitness */
 if (currentFunctionValue == fitnessVector[index[(int)_mu]]) {
   sigma *= exp(0.2+_sigmaCumulationFactor/_dampFactor);
   if (_k->_verbosity >= KORALI_DETAILED) {
     fprintf(stderr, "[Korali] Warning: sigma increased due to equal function values.\n");
   }
 }

 /*
 size_t horizon = 10 + ceil(3*10*_k->_problem->N/_s);
 double min = std::numeric_limits<double>::max();
 double max = std::numeric_limits<double>::min();
 for(size_t i = 0; (i < horizon) && (_currentGeneration - i >= 0); i++) {
    if ( histFuncValues[_currentGeneration-i] < min ) min = histFuncValues[_currentGeneration];
    if ( histFuncValues[_currentGeneration-i] > max ) max = histFuncValues[_currentGeneration];
 }
 if (max-min < 1e-12) {
   sigma *= exp(0.2+_sigmaCumulationFactor/_dampFactor);
   if (_k->_verbosity >= KORALI_DETAILED) {
     fprintf(stderr, "[Korali] Warning: sigma increased due to equal histrocial function values.\n");
   }
 }
 */

}

void CMAES::adaptC(int hsig)
{

 if (_covarianceMatrixLearningRate != 0.0)
 {
  /* definitions for speeding up inner-most loop */
  double ccov1 = std::min(_covarianceMatrixLearningRate * (1./_muCovariance) * (_isdiag ? (_k->_problem->N+1.5) / 3. : 1.), 1.);
  double ccovmu = std::min(_covarianceMatrixLearningRate * (1-1./_muCovariance) * (_isdiag ? (_k->_problem->N+1.5) / 3. : 1.), 1.-ccov1);
  double sigmasquare = sigma * sigma;

  flgEigensysIsUptodate = false;

  /* update covariance matrix */
  for (size_t i = 0; i < _k->_problem->N; ++i)
   for (size_t j = _isdiag ? i : 0; j <= i; ++j) {
   C[i][j] = (1 - ccov1 - ccovmu) * C[i][j] + ccov1 * (rgpc[i] * rgpc[j] + (1-hsig)*_cumulativeCovariance*(2.-_cumulativeCovariance) * C[i][j]);
   for (size_t k = 0; k < _mu; ++k) C[i][j] += ccovmu * _muWeights[k] * (samplePopulation[index[k]*_k->_problem->N + i] - rgxold[i]) * (samplePopulation[index[k]*_k->_problem->N + j] - rgxold[j]) / sigmasquare;
   }

  /* update maximal and minimal diagonal value */
  maxdiagC = mindiagC = C[0][0];
  for (size_t i = 1; i < _k->_problem->N; ++i) {
  if (maxdiagC < C[i][i]) maxdiagC = C[i][i];
  else if (mindiagC > C[i][i])  mindiagC = C[i][i];
  }
 } /* if ccov... */
}

bool CMAES::checkTermination()
{
 double fac;

 bool terminate = false;

 if ( _isTermCondFitness && (_currentGeneration > 1) && (bestEver >= _termCondFitness) )
 {
  terminate = true;
  sprintf(terminationReason, "Fitness Value (%7.2e) < (%7.2e)",  fitnessVector[index[0]], _termCondFitness);
 }

  double range = fabs(currentFunctionValue - prevFunctionValue);
  if ( _isTermCondFitnessDiffThreshold && (_currentGeneration > 2) && (range <= _termCondFitnessDiffThreshold) )
 {
  terminate = true;
  sprintf(terminationReason, "Function value differences (%7.2e) < (%7.2e)",  range, _termCondFitnessDiffThreshold);
 }

 size_t cTemp = 0;
 for(size_t i=0; i<_k->_problem->N; ++i) {
  cTemp += (sigma * sqrt(C[i][i]) < _termCondMinDeltaX) ? 1 : 0;
  cTemp += (sigma * rgpc[i] < _termCondMinDeltaX) ? 1 : 0;
 }

 if ( _isTermCondMinDeltaX && (cTemp == 2*_k->_problem->N) ) {
  terminate = true;
  sprintf(terminationReason, "Object variable changes < %7.2e", _termCondMinDeltaX);
 }

 for(size_t i=0; i<_k->_problem->N; ++i)
   if ( _isTermCondTolUpXFactor && (sigma * sqrt(C[i][i]) > _termCondTolUpXFactor * _k->_problem->_variables[i]->_initialStdDev) )
   {
     terminate = true;
     sprintf(terminationReason, "Standard deviation increased by more than %7.2e, larger initial standard deviation recommended \n", _termCondTolUpXFactor);
     break;
   }

  if ( _isTermCondCovCond && (maxEW >= minEW * _termCondCovCond) )
  {
    terminate = true;
    sprintf(terminationReason, "Maximal condition number %7.2e reached. maxEW=%7.2e, minEig=%7.2e, maxdiagC=%7.2e, mindiagC=%7.2e\n",
      _termCondCovCond, maxEW, minEW, maxdiagC, mindiagC);
  }

  size_t iAchse = 0;
  size_t iKoo = 0;
  if ( !_isdiag ) //TODO add isTermCond
  {
    for (iAchse = 0; iAchse < _k->_problem->N; ++iAchse)
    {
    fac = 0.1 * sigma * rgD[iAchse];
    for (iKoo = 0; iKoo < _k->_problem->N; ++iKoo){
      if (rgxmean[iKoo] != rgxmean[iKoo] + fac * B[iKoo][iAchse])
      break;
    }
    if (iKoo == _k->_problem->N)
    {
      terminate = true;
      sprintf(terminationReason, "Standard deviation 0.1*%7.2e in principal axis %ld without effect.", fac/0.1, iAchse);
      break;
    } /* if (iKoo == _k->_problem->N) */
  } /* for iAchse    */
 } /* if _isdiag */

 /* Component of rgxmean is not changed anymore */
 for (iKoo = 0; iKoo < _k->_problem->N; ++iKoo)
 {
  if (rgxmean[iKoo] == rgxmean[iKoo] + 0.2*sigma*sqrt(C[iKoo][iKoo]) ) //TODO: add _isTermCond..
  {
   /* C[iKoo][iKoo] *= (1 + _covarianceMatrixLearningRate); */
   /* flg = 1; */
   terminate = true;
   sprintf(terminationReason, "Standard deviation 0.2*%7.2e in coordinate %ld without effect.", sigma*sqrt(C[iKoo][iKoo]), iKoo);
   break;
  }

 } /* for iKoo */

 if( _isTermCondMaxFitnessEvaluations && (countevals >= _termCondMaxFitnessEvaluations))
 {
  terminate = true;
  sprintf(terminationReason, "Conducted %lu function evaluations >= (%lu).", countevals, _termCondMaxFitnessEvaluations);
 }

 if( _isTermCondMaxGenerations && (_currentGeneration >= _termCondMaxGenerations) )
 {
  terminate = true;
  sprintf(terminationReason, "Maximum number of Generations reached (%lu).", _termCondMaxGenerations);
 }

 return terminate;
}

void CMAES::updateEigensystem(int flgforce)
{
 if(flgforce == 0 && flgEigensysIsUptodate) return;
 /* if(_currentGeneration % _covarianceEigenEvalFreq == 0) return; */

 eigen(_k->_problem->N, C, rgD, B);

 /* find largest and smallest eigenvalue, they are supposed to be sorted anyway */
 minEW = doubleRangeMin(rgD, _k->_problem->N);
 maxEW = doubleRangeMax(rgD, _k->_problem->N);

 for (size_t i = 0; i < _k->_problem->N; ++i) rgD[i] = sqrt(rgD[i]);

 flgEigensysIsUptodate = true;
}

void CMAES::eigen(size_t size, double **C, double *diag, double **Q) const
{
 double* data = (double*) malloc (sizeof(double) * size * size);

 for (size_t i = 0; i <  size; i++)
 for (size_t j = 0; j <= i; j++)
 {
  data[i*size + j] = C[i][j];
  data[j*size + i] = C[i][j];
 }

 gsl_matrix_view m  = gsl_matrix_view_array (data, size, size);
 gsl_vector *eval = gsl_vector_alloc (size);
 gsl_matrix *evec = gsl_matrix_alloc (size, size);
 gsl_eigen_symmv_workspace * w =  gsl_eigen_symmv_alloc (size);
 gsl_eigen_symmv (&m.matrix, eval, evec, w);
 gsl_eigen_symmv_free (w);
 gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);

 for (size_t i = 0; i < size; i++)
 {
  gsl_vector_view evec_i = gsl_matrix_column (evec, i);
  for (size_t j = 0; j < size; j++) Q[j][i] = gsl_vector_get (&evec_i.vector, j);
 }

 for (size_t i = 0; i < size; i++) diag[i] = gsl_vector_get (eval, i);

 gsl_vector_free (eval);
 gsl_matrix_free (evec);
 free(data);
}

size_t CMAES::maxIdx(const double *rgd, size_t len) const
{
 size_t res = 0;
 for(size_t i = 1; i < len; i++)
  if(rgd[i] > rgd[res]) res = i;
 return res;
}

size_t CMAES::minIdx(const double *rgd, size_t len) const
{
 size_t res = 0;
 for(size_t i = 1; i < len; i++)
  if(rgd[i] < rgd[res]) res = i;
 return res;
}

void CMAES::sort_index(const double *fitnessVector, size_t *index, size_t n) const
{
  // initialize original index locations
  std::iota(index, index+n, (size_t) 0);

  // sort indexes based on comparing values in v
  std::sort( index, index+n, [fitnessVector](size_t i1, size_t i2) {return fitnessVector[i1] > fitnessVector[i2];} ); //descending

}

double CMAES::doubleRangeMax(const double *rgd, size_t len) const
{
 double max = rgd[0];
 for (size_t i = 1; i < len; i++)
  max = (max < rgd[i]) ? rgd[i] : max;
 return max;
}

double CMAES::doubleRangeMin(const double *rgd, size_t len) const
{
 double min = rgd[0];
 for (size_t i = 1; i < len; i++)
  min = (min > rgd[i]) ? rgd[i] : min;
 return min;
}


void CMAES::printGeneration() const
{
 if (_currentGeneration % _k->_outputFrequency != 0) return;

 if (_k->_verbosity >= KORALI_NORMAL)
   printf("--------------------------------------------------------------------\n");

 if (_k->_verbosity >= KORALI_MINIMAL)
   printf("[Korali] Generation %ld - Duration: %fs (Total Elapsed Time: %fs)\n", _currentGeneration, std::chrono::duration<double>(t1-t0).count(), std::chrono::duration<double>(t1-startTime).count());

 if (_k->_verbosity >= KORALI_NORMAL)
 {
  printf("[Korali] Sigma:                        %+6.3e\n", sigma);
  printf("[Korali] Current Function Value: Min = %+6.3e - Best = %+6.3e\n", currentFunctionValue, bestEver);
  printf("[Korali] Diagonal Covariance:    Min = %+6.3e -  Max = %+6.3e\n", mindiagC, maxdiagC);
  printf("[Korali] Covariance Eigenvalues: Min = %+6.3e -  Max = %+6.3e\n", minEW, maxEW);
 }

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Variable = (MeanX, BestX):\n");
  for (size_t i = 0; i < _k->_problem->N; i++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_problem->_variables[i]->_name.c_str(), rgxmean[i], rgxbestever[i]);

  printf("[Korali] Covariance Matrix:\n");
  for (size_t i = 0; i < _k->_problem->N; i++)
  {
   for (size_t j = 0; j <= i; j++) printf("   %+6.3e  ",C[i][j]);
   printf("\n");
  }

  printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
  printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
 }
}

void CMAES::printFinal() const
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("--------------------------------------------------------------------\n");
    printf("[Korali] CMA-ES Finished\n");
    printf("[Korali] Optimum (%s) found: %e\n", _objective.c_str(), bestEver);
    printf("[Korali] Optimum (%s) found at:\n", _objective.c_str());
    for (size_t i = 0; i < _k->_problem->N; i++) printf("         %s = %+6.3e\n", _k->_problem->_variables[i]->_name.c_str(), rgxbestever[i]);
    printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
    printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
    printf("[Korali] Stopping Criterium: %s\n", terminationReason);
    printf("[Korali] Total Elapsed Time: %fs\n", std::chrono::duration<double>(endTime-startTime).count());
 }
}
