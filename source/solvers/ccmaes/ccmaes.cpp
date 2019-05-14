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

CCMAES::CCMAES(nlohmann::json& js) : Korali::Solver::Base::Base(js)
{
 setConfiguration(js);

 // Allocating Memory
 _samplePopulation =  (double*) calloc (sizeof(double), _k->_problem->N*_s);
 rgpc = (double*) calloc (sizeof(double), _k->_problem->N);
 rgps = (double*) calloc (sizeof(double), _k->_problem->N);
 rgdTmp = (double*) calloc (sizeof(double), _k->_problem->N);
 rgBDz = (double*) calloc (sizeof(double), _k->_problem->N);
 rgxmean = (double*) calloc (sizeof(double), _k->_problem->N);
 rgxold = (double*) calloc (sizeof(double), _k->_problem->N);
 rgxbestever = (double*) calloc (sizeof(double), _k->_problem->N);
 rgD = (double*) calloc (sizeof(double), _k->_problem->N);
 curBestVector = (double *) calloc (sizeof(double), _k->_problem->N);

 rgFuncValue = (double*) calloc (sizeof(double), _s);
 histFuncValues = (double*) calloc (sizeof(double), _maxGenenerations+1);
 index = (size_t *) calloc (sizeof(size_t*), _s);

 C = (double**) calloc (sizeof(double*), _k->_problem->N);
 B = (double**) calloc (sizeof(double*), _k->_problem->N);
 for (size_t i = 0; i < _k->_problem->N; i++) C[i] = (double*) calloc (sizeof(double), _k->_problem->N);
 for (size_t i = 0; i < _k->_problem->N; i++) B[i] = (double*) calloc (sizeof(double), _k->_problem->N);

 _initializedSample = (bool*) calloc (_s, sizeof(bool));
 _fitnessVector = (double*) calloc (sizeof(double), _s);

 // Initializing Gaussian Generator
 _gaussianGenerator = new Variable::Gaussian(0.0, 1.0, _k->_seed++);

 // Checking Covariance Matrix Evaluation Frequency
 if (_diagonalCovarianceMatrixEvalFrequency == 0)  _diagonalCovarianceMatrixEvalFrequency = 2 + 100. * _k->_problem->N / sqrt((double)_s);
 if (_diagonalCovarianceMatrixEvalFrequency < 1)
 { fprintf( stderr, "[Korali] Error: Matrix covariance evaluation frequency is less than 1 (%lu)\n", _diagonalCovarianceMatrixEvalFrequency); exit(-1); }

 // Setting Sigma Cumulation Factor
 if (_sigmaCumulationFactor > 0) _sigmaCumulationFactor *= (_muEffective + 2.0) / (_k->_problem->N + _muEffective + 3.0);
 if (_sigmaCumulationFactor <= 0 || _sigmaCumulationFactor >= 1) _sigmaCumulationFactor = (_muEffective + 2.) / (_k->_problem->N + _muEffective + 3.0);

 // Setting Damping Factor

 if (_dampFactor < 0) _dampFactor = 1;
 _dampFactor = _dampFactor* (1 + 2*std::max(0.0, sqrt((_muEffective-1.0)/(_k->_problem->N+1.0)) - 1))  /* basic factor */
  * std::max(0.3, 1. - (double)_k->_problem->N / (1e-6+std::min(_maxGenenerations, _maxFitnessEvaluations/_s))) /* modification for short runs */
  + _sigmaCumulationFactor; /* minor increment */

 // Setting Cumulative Covariance

 if (_cumulativeCovariance <= 0 || _cumulativeCovariance> 1)  _cumulativeCovariance = 4. / (_k->_problem->N + 4);

 // Set covariance Matrix Learning Rate

 double l1 = 2. / ((_k->_problem->N+1.4142)*(_k->_problem->N+1.4142));
 double l2 = (2.*_muEffective-1.) / ((_k->_problem->N+2.)*(_k->_problem->N+2.)+_muEffective);
 l2 = (l2 > 1) ? 1 : l2;
 l2 = (1./_muCovariance) * l1 + (1.-1./_muCovariance) * l2;

 if (_covarianceMatrixLearningRate >= 0) _covarianceMatrixLearningRate *= l2;
 if (_covarianceMatrixLearningRate < 0 || _covarianceMatrixLearningRate > 1)  _covarianceMatrixLearningRate = l2;

 // Setting eigensystem evaluation Frequency

 if( _covarianceEigenEvalFreq < 0.0) _covarianceEigenEvalFreq = 1.0/_covarianceMatrixLearningRate/((double)_k->_problem->N)/10.0;

 double trace = 0.0;
 for (size_t i = 0; i < _k->_problem->N; ++i) trace += _k->_problem->_parameters[i]->_initialStdDev*_k->_problem->_parameters[i]->_initialStdDev;
 sigma = sqrt(trace/_k->_problem->N); /* _muEffective/(0.2*_muEffective+sqrt(_k->_problem->N)) * sqrt(trace/_k->_problem->N); */

 flgEigensysIsUptodate = true;

 countevals = 0;
 countinfeasible = 0;
 bestEver = 0.0;

 for (size_t i = 0; i < _k->_problem->N; ++i)
 {
  B[i][i] = 1.0;
  C[i][i] = rgD[i] = _k->_problem->_parameters[i]->_initialStdDev * sqrt(_k->_problem->N / trace);
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
   if(_k->_problem->_parameters[i]->_initialValue < _k->_problem->_parameters[i]->_lowerBound || _k->_problem->_parameters[i]->_initialValue > _k->_problem->_parameters[i]->_upperBound)
    fprintf(stderr,"[Korali] Warning: Initial Value (%.4f) for \'%s\' is out of bounds (%.4f-%.4f).\n", _k->_problem->_parameters[i]->_initialValue, _k->_problem->_parameters[i]->_name.c_str(), _k->_problem->_parameters[i]->_lowerBound, _k->_problem->_parameters[i]->_upperBound);
   rgxmean[i] = rgxold[i] = _k->_problem->_parameters[i]->_initialValue;
 }

 // CCMA-ES variaables
  _numConstraints = _k->_fconstraints.size();
 viabilityBounds = new double [_numConstraints]; 
 std::fill_n( viabilityBounds, _numConstraints, -1);
 
 sucRates = (double*) calloc (sizeof(double), _numConstraints); 
 std::fill_n( sucRates, _numConstraints, 0.5);

 maxnumviolations        = -1;
 numviolations           = (size_t*) calloc (sizeof(size_t), _s); 
 std::fill_n(numviolations, _s, -1);
 
 viabilityImprovement    = (bool*) calloc (sizeof(bool), _s);
 
 viabilityIndicator    = (bool**) calloc (sizeof(bool*), _numConstraints);
 constraintEvaluations = (double**) calloc (sizeof(double*), _numConstraints);
 for (size_t i = 0; i < _numConstraints; ++i) viabilityIndicator[i]    = (bool*) calloc (sizeof(bool), _s);
 for (size_t i = 0; i < _numConstraints; ++i) constraintEvaluations[i] = (double*) calloc (sizeof(double), _s);

 v = (double**) calloc (sizeof(double*), _numConstraints);
 for (size_t i = 0; i < _numConstraints; ++i) constraintEvaluations[i] = (double*) calloc (sizeof(double), _s);
 
 Z   = (double**) calloc (sizeof(double*), _s);
 BDZ = (double**) calloc (sizeof(double*), _s);
 for (size_t i = 0; i < _k->_problem->N; i++) Z[i]   = (double*) calloc (sizeof(double), _k->_problem->N);
 for (size_t i = 0; i < _k->_problem->N; i++) BDZ[i] = (double*) calloc (sizeof(double), _k->_problem->N);
 // TODO: what else?


 // If state is defined:
 if (isDefined(js, {"State"}))
 {
  setState(js);
  js.erase("State");
 }


}

CCMAES::~CCMAES()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json CCMAES::getConfiguration()
{
 auto js = this->Korali::Solver::Base::getConfiguration();

 js["Method"] = "CCMA-ES";

 js["Num Samples"]             = _s;
 js["Num Viability Samples"]   = _via_s;
 js["Current Generation"]      = _currentGeneration;
 js["Sigma Cumulation Factor"] = _sigmaCumulationFactor;
 js["Damp Factor"]             = _dampFactor;
 js["Objective"]               = _objective;

 js["Mu"]["Value"]      = _mu;
 js["Mu"]["Type"]       = _muType;
 js["Mu"]["Covariance"] = _muCovariance;
 js["Mu"]["Effective"]  = _muEffective;
 for (size_t i = 0; i < _mu; i++) js["Mu"]["Weights"] += _muWeights[i];

 js["Covariance Matrix"]["Eigenvalue Evaluation Frequency"] = _covarianceEigenEvalFreq;
 js["Covariance Matrix"]["Cumulative Covariance"]           = _cumulativeCovariance;
 js["Covariance Matrix"]["Learning Rate"]                   = _covarianceMatrixLearningRate;
 js["Covariance Matrix"]["Enable Diagonal Update"]          = _enablediag;

 js["Termination Criteria"]["Max Generations"]          = _maxGenenerations;
 js["Termination Criteria"]["Min Fitness"]              = _stopMinFitness;
 js["Termination Criteria"]["Max Model Evaluations"]    = _maxFitnessEvaluations;
 js["Termination Criteria"]["Fitness Diff Threshold"]   = _stopFitnessDiffThreshold;
 js["Termination Criteria"]["Min DeltaX"]               = _stopMinDeltaX;
 js["Termination Criteria"]["Max Standard Deviation"]   = _stopTolUpXFactor;
 js["Termination Criteria"]["Max Condition Covariance"] = _stopCovCond;
 js["Termination Criteria"]["Ignore"]                   = _ignorecriteria;

 // State Variables
 js["State"]["Current Generation"]       = _currentGeneration;
 js["State"]["Fitness Sign"]             = _fitnessSign;
 js["State"]["Sigma"]                    = sigma;
 js["State"]["BestEverFunctionValue"]    = bestEver;
 js["State"]["CurrentBestFunctionValue"] = currentFunctionValue;
 js["State"]["PrevFunctionValue"]        = prevFunctionValue;
 js["State"]["MaxDiagonalCovariance"]    = maxdiagC;
 js["State"]["MinDiagonalCovariance"]    = mindiagC;
 js["State"]["MaxEigenvalue"]            = maxEW;
 js["State"]["MinEigenvalue"]            = minEW;
 js["State"]["EigenSystemUpToDate"]      = flgEigensysIsUptodate;
 js["State"]["EvaluationCount"]          = countevals;
 js["State"]["InfeasibleCount"]          = countinfeasible;
 js["State"]["ConjugateEvolutionPathL2"] = psL2;

 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["CurrentMeanVector"]      += rgxmean[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["PreviousMeanVector"]     += rgxold[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["BestEverVector"]         += rgxbestever[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["CurrentBestVector"]      += curBestVector[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["AxisLengths"]            += rgD[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["EvolutionPath"]          += rgpc[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["ConjugateEvolutionPath"] += rgps[i];
 for (size_t i = 0; i < _mu; i++)   js["State"]["Index"]                  += index[i];
 for (size_t i = 0; i < _mu; i++)   js["State"]["FunctionValues"]         += rgFuncValue[i];

 for (size_t i = 0; i < _k->_problem->N; i++) for (size_t j = 0; j < _k->_problem->N; j++) js["State"]["CovarianceMatrix"][i][j] = C[i][j];
 for (size_t i = 0; i < _k->_problem->N; i++) for (size_t j = 0; j < _k->_problem->N; j++) js["State"]["EigenMatrix"][i][j] = B[i][j];
 for (size_t i = 0; i < _s; i++) for (size_t j = 0; j < _k->_problem->N; j++)    js["State"]["Samples"][i][j] = _samplePopulation[i*_k->_problem->N + j];

 // CCMA-ES Variables
 js["Global Success Rate"]          = globalSucRate;
 js["Target Success Rate"]          = _targetSucRate;
 js["Adaption Size"]                = _beta;
 js["Max Adaptions"]                = _maxAdaptions;
 js["Max Resamplings"]              = _maxResamplings;
 js["Normal Vector Learning Rate"]  = _cv;
 js["Global Success Learning Rate"] = _cp;
 
 // CCMA-ES States
 for (size_t i = 0; i < _numConstraints; ++i) js["State"]["Success Rates"] = sucRates[i];
 for (size_t i = 0; i < _numConstraints; ++i) js["State"]["Viability Boundaries"][i] = viabilityBounds[i];
 
 // TODO: some things are missing here, or we remove 'unnecessaty' states

 return js;
}


void CCMAES::setConfiguration(nlohmann::json& js)
{
 this->Korali::Solver::Base::setConfiguration(js);

 _s                             = consume(js, { "Num Samples" }, KORALI_NUMBER);
 _via_s                         = consume(js, { "Num Viability Samples" }, KORALI_NUMBER, std::to_string(2));
 _currentGeneration             = consume(js, { "Current Generation" }, KORALI_NUMBER, std::to_string(0));
 _sigmaCumulationFactor         = consume(js, { "Sigma Cumulation Factor" }, KORALI_NUMBER, std::to_string(-1));
 _dampFactor                    = consume(js, { "Damp Factor" }, KORALI_NUMBER, std::to_string(-1));
 _objective                     = consume(js, { "Objective" }, KORALI_STRING, "Maximize");

 _fitnessSign = 0;
 if(_objective == "Maximize") _fitnessSign = 1;
 if(_objective == "Minimize") _fitnessSign = -1;
 if(_fitnessSign == 0)  { fprintf( stderr, "[Korali] Error: Invalid setting for Objective: %s\n", _objective.c_str()); exit(-1); }

 _mu                            = consume(js, { "Mu", "Value" }, KORALI_NUMBER, std::to_string(ceil(_s / 2)));
 _muType                        = consume(js, { "Mu", "Type" }, KORALI_STRING, "Logarithmic");
 _muCovariance                  = consume(js, { "Mu", "Covariance" }, KORALI_NUMBER, std::to_string(-1));

 // Initializing Mu Weights
 _muWeights = (double *) calloc (sizeof(double), _mu);
 if (_muType == "Linear")           for (size_t i = 0; i < _mu; i++)  _muWeights[i] = _mu - i;
 if (_muType == "Equal")            for (size_t i = 0; i < _mu; i++)  _muWeights[i] = 1;
 if (_muType == "Logarithmic")      for (size_t i = 0; i < _mu; i++)  _muWeights[i] = log(_mu+1.)-log(i+1.);

 _chiN = sqrt((double) _k->_problem->N) * (1. - 1./(4.*_k->_problem->N) + 1./(21.*_k->_problem->N*_k->_problem->N));
 
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

 _covarianceEigenEvalFreq      = consume(js, { "Covariance Matrix", "Eigenvalue Evaluation Frequency" }, KORALI_NUMBER, std::to_string(-1));
 _cumulativeCovariance         = consume(js, { "Covariance Matrix", "Cumulative Covariance" }, KORALI_NUMBER, std::to_string(-1));
 _covarianceMatrixLearningRate = consume(js, { "Covariance Matrix", "Learning Rate" }, KORALI_NUMBER, std::to_string(-1));
 _enablediag                   = consume(js, { "Covariance Matrix", "Enable Diagonal Update" }, KORALI_BOOLEAN, "false");
 _maxGenenerations             = consume(js, { "Termination Criteria", "Max Generations" }, KORALI_NUMBER, std::to_string(1000));
 _stopMinFitness               = consume(js, { "Termination Criteria", "Min Fitness" }, KORALI_NUMBER, std::to_string(-std::numeric_limits<double>::max()));
 _maxFitnessEvaluations        = consume(js, { "Termination Criteria", "Max Model Evaluations" }, KORALI_NUMBER, std::to_string(std::numeric_limits<size_t>::max()));
 _stopFitnessDiffThreshold     = consume(js, { "Termination Criteria", "Fitness Diff Threshold" }, KORALI_NUMBER, std::to_string(1e-9));
 _stopMinDeltaX                = consume(js, { "Termination Criteria", "Min DeltaX" }, KORALI_NUMBER, std::to_string(0.0));
 _stopTolUpXFactor             = consume(js, { "Termination Criteria", "Max Standard Deviation" }, KORALI_NUMBER, std::to_string(1e18));
 _stopCovCond                  = consume(js, { "Termination Criteria", "Max Condition Covariance" }, KORALI_NUMBER, std::to_string(std::numeric_limits<double>::max()));
 _ignorecriteria               = consume(js, { "Termination Criteria", "Ignore" }, KORALI_STRING, "Max Condition Covariance");

 
 globalSucRate   = consume(js, { "Global Success Rate" }, KORALI_NUMBER, std::to_string(0.44));
 _targetSucRate  = consume(js, { "Target Success Rate" }, KORALI_NUMBER, std::to_string(2./11.));
 _beta           = consume(js, { "Adaption Size" }, KORALI_NUMBER, std::to_string(0.1/(_s+2.)));
 _maxAdaptions   = consume(js, { "Max Adaptions" }, KORALI_NUMBER, std::to_string(1e6));
 _maxResamplings = consume(js, { "Max Resamplings" }, KORALI_NUMBER, std::to_string(1e6));
 _cv             = consume(js, { "Normal Vector Learning Rate" }, KORALI_NUMBER, std::to_string(1.0/(_s+2.)));
 _cp             = consume(js, { "Global Success Learning Rate" }, KORALI_NUMBER, std::to_string(1.0/12.0));
 
}


void CCMAES::setState(nlohmann::json& js)
{
 this->Korali::Solver::Base::setState(js);

 _currentGeneration    = js["State"]["Current Generation"];
 _fitnessSign          = js["State"]["Fitness Sign"];
 sigma                 = js["State"]["Sigma"];
 bestEver              = js["State"]["BestEverFunctionValie"];
 currentFunctionValue  = js["State"]["CurrentBestFunctionValue"];
 prevFunctionValue     = js["State"]["PrevFunctionValue"];
 maxdiagC              = js["State"]["MaxDiagonalCovariance"];
 mindiagC              = js["State"]["MinDiagonalCovariance"];
 maxEW                 = js["State"]["MaxEigenvalue"];
 minEW                 = js["State"]["MinEigenvalue"] ;
 flgEigensysIsUptodate = js["State"]["EigenSystemUpToDate"];
 countevals            = js["State"]["EvaluationCount"];
 countinfeasible       = js["State"]["InfeasibleCount"];

 for (size_t i = 0; i < _k->_problem->N; i++) rgxmean[i]       = js["State"]["CurrentMeanVector"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) rgxold[i]        = js["State"]["PreviousMeanVector"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) rgxbestever[i]   = js["State"]["BestEverVector"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) curBestVector[i] = js["State"]["CurrentBestVector"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) index[i]         = js["State"]["Index"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) rgD[i]           = js["State"]["AxisLengths"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) rgpc[i]          = js["State"]["CumulativeCovariance"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) rgFuncValue[i]   = js["State"]["FunctionValues"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) for (size_t j = 0; j < _k->_problem->N; j++) C[i][j] = js["State"]["CovarianceMatrix"][i][j];
 for (size_t i = 0; i < _k->_problem->N; i++) for (size_t j = 0; j < _k->_problem->N; j++) B[i][j] = js["State"]["EigenMatrix"][i][j];
 for (size_t i = 0; i < _s; i++) for (size_t j = 0; j < _k->_problem->N; j++) _samplePopulation[i*_k->_problem->N + j] = js["State"]["Samples"][i][j];

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void CCMAES::run()
{
 if (_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Starting CMA-ES (Objective: %s).\n", _objective.c_str());

 startTime = std::chrono::system_clock::now();
 _k->saveState();

 while(!checkTermination())
 {
 t0 = std::chrono::system_clock::now();
 prepareGeneration();
 evaluateSamples();
 updateDistribution(_fitnessVector);
 handleViabilityConstraints();

 t1 = std::chrono::system_clock::now();
 printGeneration();

 _k->saveState();
 }

 endTime = std::chrono::system_clock::now();

 printFinal();

}


void CCMAES::evaluateSamples() 
{
  while (_finishedSamples < _s)
  {
    for (size_t i = 0; i < _s; i++) if (_initializedSample[i] == false)
    {
      _initializedSample[i] = true;
      _k->_conduit->evaluateSample(_samplePopulation, i);
    }
    _k->_conduit->checkProgress();
  }
}


void CCMAES::processSample(size_t sampleId, double fitness)
{
 _fitnessVector[sampleId] = _fitnessSign*fitness;
 _finishedSamples++;
}


void CCMAES::evaluateConstraints() //TODO: maybe we can parallelize constraint evaluations (DW)
{
  for(size_t c = 0; c < _numConstraints; ++c)
  {
    for(size_t i = 0; i < _s; ++i)
    {
      if(numviolations[i] != 0) // initialized with -1
      {
        numviolations[i] = 0;
        countcevals++;
        constraintEvaluations[c][i] = _k->_fconstraints[c]( _samplePopulation+i*_k->_problem->N, _k->_problem->N );
        if( constraintEvaluations[c][i] > viabilityBounds[c] ) { viabilityIndicator[c][i]=1; numviolations[i]++; }
        else viabilityIndicator[c][i] = 0;
      }
      if (numviolations[i] > maxnumviolations) maxnumviolations = numviolations[i];
    }
  }
}


void CCMAES::updateViabilityBoundaries()
{
   /* TODO: check if this part is needed 
   //TODO: frgxmean = _k->_conduit->_evaluateSample(rgxmean) (funcionality not yet implemented! (DW))
   fviability = frgxmean + 0.5*(frgxold-frgxmean);
   frgxold    = frgxmean;
   */

 for(size_t c = 0; c < _numConstraints; ++c)
 {
  double maxviolation = std::numeric_limits<double>::min();
  for(size_t i = 0; i < _s; ++i) if (constraintEvaluations[c][index[i]] > maxviolation) //TODO: or _mu instead of _s (original) (DW)??
   maxviolation = constraintEvaluations[c][index[i]];
     
  viabilityBounds[c] = std::max( 0.0, std::min(viabilityBounds[c], 0.5*(maxviolation + viabilityBounds[c])) );
 }
}


bool CCMAES::isFeasible(size_t sampleIdx) const
{
 for (size_t d = 0; d < _k->_problem->N; ++d)
  if (_samplePopulation[ sampleIdx*_k->_problem->N+d ] < _k->_problem->_parameters[d]->_lowerBound || _samplePopulation[ sampleIdx*_k->_problem->N+d ] > _k->_problem->_parameters[d]->_upperBound) return false;
 return true;
}


void CCMAES::prepareGeneration()
{
 int flgdiag = doDiagUpdate();

 /* calculate eigensystem  */
 if (!flgEigensysIsUptodate) {
  if (!flgdiag)
   updateEigensystem(0);
  else {
   for (size_t d = 0; d < _k->_problem->N; ++d) rgD[d] = sqrt(C[d][d]);
   minEW = doubleRangeMin(rgD, _k->_problem->N) * doubleRangeMin(rgD, _k->_problem->N);
   maxEW = doubleRangeMax(rgD, _k->_problem->N) * doubleRangeMin(rgD, _k->_problem->N);
   flgEigensysIsUptodate = true;
  }
 }

 for (size_t iNk = 0; iNk < _s; ++iNk) 
 {
     sampleSingle(iNk);
     while( !isFeasible( iNk )) { countinfeasible++; sampleSingle(iNk); } //TODO: maybe add exit criterion
 }

 _currentGeneration++;

 _finishedSamples = 0;
 for (size_t i = 0; i < _s; i++) _initializedSample[i] = false;
}


void CCMAES::sampleSingle(size_t sampleIdx) 
{
  int flgdiag = doDiagUpdate();
  
  /* generate scaled random vector (D * z) */
  for (size_t d = 0; d < _k->_problem->N; ++d)
  {
   Z[sampleIdx][d] = _gaussianGenerator->getRandomNumber();
   if (flgdiag) { 
     BDZ[sampleIdx][d] = rgD[d] * Z[sampleIdx][d];
     _samplePopulation[sampleIdx * _k->_problem->N + d] = rgxmean[d] + sigma * BDZ[sampleIdx][d]; 
   }
   else rgdTmp[d] = rgD[d] * Z[sampleIdx][d];
  }

  if (!flgdiag)
   for (size_t d = 0; d < _k->_problem->N; ++d) {
    BDZ[sampleIdx][d] = 0.0;
    for (size_t e = 0; e < _k->_problem->N; ++e) BDZ[sampleIdx][d] += B[d][e] * rgdTmp[d];
    _samplePopulation[sampleIdx * _k->_problem->N + d] = rgxmean[d] + sigma * BDZ[sampleIdx][d];
  }
}


void CCMAES::updateDistribution(const double *fitnessVector)
{
 int flgdiag = doDiagUpdate();
 countevals += _s;

 /* Generate index */
 sort_index(fitnessVector, index, _s);

 /* assign function values */
 for (size_t i = 0; i < _s; i++) rgFuncValue[i] = fitnessVector[i];

 /* update function value history */
 prevFunctionValue = currentFunctionValue;

 /* update current best */
 currentFunctionValue = fitnessVector[index[0]];
 for (size_t d = 0; d < _k->_problem->N; ++d) curBestVector[d] = _samplePopulation[index[0]*_k->_problem->N + d];
 
 /* update xbestever */
 if (currentFunctionValue > bestEver || _currentGeneration == 1)
 {
  prevBest = bestEver;
  bestEver = currentFunctionValue;
  for (size_t d = 0; d < _k->_problem->N; ++d) rgxbestever[d] = curBestVector[d];
 }

 histFuncValues[_currentGeneration] = bestEver;

 /* calculate rgxmean and rgBDz */
 for (size_t d = 0; d < _k->_problem->N; ++d) {
   rgxold[d] = rgxmean[d];
   rgxmean[d] = 0.;
   for (size_t iNk = 0; iNk < _mu; ++iNk)
     rgxmean[d] += _muWeights[iNk] * _samplePopulation[index[iNk]*_k->_problem->N + d];
   
   rgBDz[d] = (rgxmean[d] - rgxold[d])/sigma;
 }

 /* calculate z := D^(-1) * B^(T) * rgBDz into rgdTmp */
 for (size_t d = 0; d < _k->_problem->N; ++d) {
  double sum = 0.0;
  if (flgdiag) sum = rgBDz[d];
  else for (size_t e = 0; e < _k->_problem->N; ++e) sum += B[e][d] * rgBDz[e]; /* B^(T) * rgBDz ( iterating B[e][d] = B^(T) ) */
  
  rgdTmp[d] = sum / rgD[d]; /* D^(-1) */
 }

 psL2 = 0.0;

 /* cumulation for sigma (ps) using B*z */
 for (size_t d = 0; d < _k->_problem->N; ++d) {
    double sum = 0.0;
    if (flgdiag) sum = rgdTmp[d];
    else for (size_t e = 0; e < _k->_problem->N; ++e) sum += B[d][e] * rgdTmp[d];

    rgps[d] = (1. - _sigmaCumulationFactor) * rgps[d] + sqrt(_sigmaCumulationFactor * (2. - _sigmaCumulationFactor) * _muEffective) * sum;
 
    /* calculate norm(ps)^2 */
    psL2 += rgps[d] * rgps[d];
 }

 int hsig = sqrt(psL2) / sqrt(1. - pow(1.-_sigmaCumulationFactor, 2*_currentGeneration)) / _chiN  < 1.4 + 2./(_k->_problem->N+1);

 /* cumulation for covariance matrix (pc) using B*D*z~_k->_problem->N(0,C) */
 for (size_t d = 0; d < _k->_problem->N; ++d)
   rgpc[d] = (1. - _cumulativeCovariance) * rgpc[d] + hsig * sqrt( _cumulativeCovariance * (2. - _cumulativeCovariance) *_muEffective ) * rgBDz[d];

 /* update of C  */
 adaptC(hsig);

 /* update of sigma */
 sigma *= exp(((sqrt(psL2)/_chiN)-1.)*_sigmaCumulationFactor/_dampFactor);



 /* TODO: viability
    - update pglobal
    - update sigma using pglobal
    - update viability boundaries
                */


 /* numerical error management */
 
 //treat maximal standnard deviations
 //TODO

 //treat minimal standard deviations
 for (size_t d = 0; d < _k->_problem->N; ++d)
   if (sigma * sqrt(C[d][d]) < _k->_problem->_parameters[d]->_minStdDevChange) {
     sigma = (_k->_problem->_parameters[d]->_minStdDevChange)/sqrt(C[d][d]) * exp(0.05+_sigmaCumulationFactor/_dampFactor);
     if (_k->_verbosity >= KORALI_DETAILED) fprintf(stderr, "[Korali] Warning: sigma increased due to minimal standard deviation.\n");
   }

 //too low coordinate axis deviations
 //TODO
 
 //treat numerical precision provblems
 //TODO

 /* Test if function values are identical, escape flat fitness */
 if (currentFunctionValue == rgFuncValue[index[(int)_mu]]) {
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


void CCMAES::adaptC(int hsig)
{
 int flgdiag = doDiagUpdate();

 if (_covarianceMatrixLearningRate != 0.0)
 {
  /* definitions for speeding up inner-most loop */
  double ccov1 = std::min(_covarianceMatrixLearningRate * (1./_muCovariance) * (flgdiag ? (_k->_problem->N+1.5) / 3. : 1.), 1.);
  double ccovmu = std::min(_covarianceMatrixLearningRate * (1-1./_muCovariance) * (flgdiag ? (_k->_problem->N+1.5) / 3. : 1.), 1.-ccov1);
  double sigmasquare = sigma * sigma;

  flgEigensysIsUptodate = false;

  /* update covariance matrix */
  for (size_t d = 0; d < _k->_problem->N; ++d)
   for (size_t e = flgdiag ? d : 0; e <= d; ++e) {
   C[d][e] = (1 - ccov1 - ccovmu) * C[d][e] + ccov1 * (rgpc[d] * rgpc[e] + (1-hsig)*_cumulativeCovariance*(2.-_cumulativeCovariance) * C[d][e]);
   for (size_t k = 0; k < _mu; ++k) C[d][e] += ccovmu * _muWeights[k] * (_samplePopulation[index[k]*_k->_problem->N + d] - rgxold[d]) * (_samplePopulation[index[k]*_k->_problem->N + e] - rgxold[e]) / sigmasquare;
   }

  /* update maximal and minimal diagonal value */
  maxdiagC = mindiagC = C[0][0];
  for (size_t d = 1; d < _k->_problem->N; ++d) {
  if (maxdiagC < C[d][d]) maxdiagC = C[d][d];
  else if (mindiagC > C[d][d])  mindiagC = C[d][d];
  }
 } /* if ccov... */
}


bool CCMAES::validateMeanAndSetRegime()
{
    for (size_t c = 0; c < _numConstraints; ++c) if ( _k->_fconstraints[c](rgxmean, _k->_problem->N) > 0.) return false;
    return true;
}


void CCMAES::updateSigmaVIE() 
{
 sigma *= exp(1.0/_dampFactor)*(globalSucRate-(_targetSucRate/(1.0-_targetSucRate))*(1-globalSucRate));
 if(_k->_verbosity >= KORALI_DETAILED && sigma > 0.3) printf("[Korali] Warning: updateSigmaVie: sigma (%f) > 0.3\n", sigma);
 // sigma = std::min(sigma, 0.2);
};


void CCMAES::handleViabilityConstraints()
{ 
 size_t initial_resampled = resampled;
 size_t initial_adaptions = adaptionsVia;

 evaluateConstraints(); 

 //for(size_t i = 0; i < _s; ++i) if(_fitnessVector[i] > fviability) viabilityImprovement[i] = true; //TODO: is this needed here?

 while( maxnumviolations != 0 )
 {

  for(size_t i = 0; i < _s; ++i) if (numviolations[i] > 0)
  {
    //update v
    for( size_t c = 0; c < _numConstraints; ++c )
    {
      double v2 = 0;
      if ( viabilityIndicator[c][i] == true )
      {
        adaptionsVia++;
        for( size_t d = 0; d < _k->_problem->N; ++d)
        {
            v[c][d] = (1.0-_cv)*v[c][d]+_cv*BDZ[i][d];
            v2 += v[c][d]*v[c][d];
        }
        for( size_t d = 0; d < _k->_problem->N; ++d)
          for( size_t e = 0; e < _k->_problem->N; ++e)
            C[d][e] = C[d][e] - ((_beta * v[c][d]*v[c][e])/(v2*numviolations[i]));

        /* in original update invBD (TODO: not needed?) */

        sucRates[i] = (1.0-_cp)*sucRates[i];
      }
      else
      {
        sucRates[i] = (1.0-_cp)*sucRates[i]+_cp/_s;
      }
    }
  }
  
  //C
  /* in original update C: not needed anymore */
  /* TODO: verify this */
  eigen(_k->_problem->N, C, rgD, B);
  
  /* in original some stopping criterion (TOLX) */

  //resample invalid points
  for(size_t i = 0; i < _s; ++i) if(numviolations[i] > 0)
  {
    do
    {
     resampled++;
     sampleSingle(i);
     if(resampled-initial_resampled > _maxResamplings)
     {
        if(_k->_verbosity >= KORALI_DETAILED) printf("[Korali] Warning: exiting VIE resampling loop, max resamplings reached.\n ");
        return;
     }

    }
    while( !isFeasible(i) );
  }
 
  evaluateConstraints();
  
  for(size_t c = 0; c < _numConstraints; ++c) if(sucRates[c] < 0.5)
  {
    globalSucRate = (1-_cp) * globalSucRate; //TODO: verify this update
    break; //TODO: verify this break
  }
  

  if(adaptionsVia - initial_adaptions > _maxAdaptions)
  {
    if(_k->_verbosity >= KORALI_DETAILED) printf("[Korali] Warning: exiting VIE adaption loop, max adaptions reached.\n ");
    return;
  }

 }//while maxnumviolations != 0
 
}


void CCMAES::handleConstraints() 
{ 

    //TODO
 
}


bool CCMAES::checkTermination()
{
 double fac;
 int flgdiag = doDiagUpdate();

 bool terminate = false;

 if (_currentGeneration > 1 && rgFuncValue[index[0]] <= _stopMinFitness && isStoppingCriteriaActive("Fitness Value") )
 {
  terminate = true;
  sprintf(_terminationReason, "Fitness Value (%7.2e) < (%7.2e)",  rgFuncValue[index[0]], _stopMinFitness);
 }

  double range = fabs(currentFunctionValue - prevFunctionValue);
  if (_currentGeneration > 2 && range <= _stopFitnessDiffThreshold && isStoppingCriteriaActive("Fitness Diff Threshold") )
 {
  terminate = true;
  sprintf(_terminationReason, "Function value differences (%7.2e) < (%7.2e)",  range, _stopFitnessDiffThreshold);
 }

 size_t cTemp = 0;
 for(size_t i=0; i<_k->_problem->N; ++i) {
  cTemp += (sigma * sqrt(C[i][i]) < _stopMinDeltaX) ? 1 : 0;
  cTemp += (sigma * rgpc[i] < _stopMinDeltaX) ? 1 : 0;
 }

 if (cTemp == 2*_k->_problem->N && isStoppingCriteriaActive("Min DeltaX") ) {
  terminate = true;
  sprintf(_terminationReason, "Object variable changes < %7.2e", _stopMinDeltaX);
 }

 for(size_t i=0; i<_k->_problem->N; ++i)
   if (sigma * sqrt(C[i][i]) > _stopTolUpXFactor * _k->_problem->_parameters[i]->_initialStdDev && isStoppingCriteriaActive("Max Standard Deviation") )
   {
     terminate = true;
     sprintf(_terminationReason, "Standard deviation increased by more than %7.2e, larger initial standard deviation recommended \n", _stopTolUpXFactor);
     break;
   }

  if (maxEW >= minEW * _stopCovCond && isStoppingCriteriaActive("Max Condition Covariance") )
  {
    terminate = true;
    sprintf(_terminationReason, "Maximal condition number %7.2e reached. maxEW=%7.2e, minEig=%7.2e, maxdiagC=%7.2e, mindiagC=%7.2e\n",
      _stopCovCond, maxEW, minEW, maxdiagC, mindiagC);
  }

  size_t iAchse = 0;
  size_t iKoo = 0;
  if (!flgdiag && isStoppingCriteriaActive("No Effect Axis") )
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
      sprintf(_terminationReason, "Standard deviation 0.1*%7.2e in principal axis %ld without effect.", fac/0.1, iAchse);
      break;
    } /* if (iKoo == _k->_problem->N) */
  } /* for iAchse    */
 } /* if flgdiag */

 /* Component of rgxmean is not changed anymore */
 for (iKoo = 0; iKoo < _k->_problem->N; ++iKoo)
 {
  if (rgxmean[iKoo] == rgxmean[iKoo] + 0.2*sigma*sqrt(C[iKoo][iKoo]) && isStoppingCriteriaActive("No Effect Standard Deviation") )
  {
   /* C[iKoo][iKoo] *= (1 + _covarianceMatrixLearningRate); */
   /* flg = 1; */
   terminate = true;
   sprintf(_terminationReason, "Standard deviation 0.2*%7.2e in coordinate %ld without effect.", sigma*sqrt(C[iKoo][iKoo]), iKoo);
   break;
  }

 } /* for iKoo */

 if(countevals >= _maxFitnessEvaluations && isStoppingCriteriaActive("Max Model Evaluations") )
 {
  terminate = true;
  sprintf(_terminationReason, "Conducted %lu function evaluations >= (%lu).", countevals, _maxFitnessEvaluations);
 }

 if(_currentGeneration >= _maxGenenerations && isStoppingCriteriaActive("Max Generations") )
 {
  terminate = true;
  sprintf(_terminationReason, "Maximum number of Generations reached (%lu).", _maxGenenerations);
 }

 return terminate;
}


void CCMAES::updateEigensystem(int flgforce)
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


void CCMAES::eigen(size_t size, double **C, double *diag, double **Q) const
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


size_t CCMAES::maxIdx(const double *rgd, size_t len) const
{
 size_t res = 0;
 for(size_t i = 1; i < len; i++)
  if(rgd[i] > rgd[res]) res = i;
 return res;
}


size_t CCMAES::minIdx(const double *rgd, size_t len) const
{
 size_t res = 0;
 for(size_t i = 1; i < len; i++)
  if(rgd[i] < rgd[res]) res = i;
 return res;
}


void CCMAES::sort_index(const double *fitnessVector, size_t *index, size_t n) const
{
  // initialize original index locations
  std::iota(index, index+n, (size_t) 0);

  // sort indexes based on comparing values in v
  std::sort( index, index+n, [fitnessVector](size_t i1, size_t i2) {return fitnessVector[i1] > fitnessVector[i2];} ); //descending

}


double CCMAES::doubleRangeMax(const double *rgd, size_t len) const
{
 double max = rgd[0];
 for (size_t i = 1; i < len; i++)
  max = (max < rgd[i]) ? rgd[i] : max;
 return max;
}


double CCMAES::doubleRangeMin(const double *rgd, size_t len) const
{
 double min = rgd[0];
 for (size_t i = 1; i < len; i++)
  min = (min > rgd[i]) ? rgd[i] : min;
 return min;
}


bool CCMAES::doDiagUpdate() const
{
 return _enablediag && (_currentGeneration % _diagonalCovarianceMatrixEvalFrequency == 0);
}


bool CCMAES::isStoppingCriteriaActive(const char *criteria) const
{
    std::string c(criteria);
    size_t found = _ignorecriteria.find(c);
    return (found==std::string::npos);
}


void CCMAES::printGeneration() const
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
  for (size_t i = 0; i < _k->_problem->N; i++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_problem->_parameters[i]->_name.c_str(), rgxmean[i], rgxbestever[i]);

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


void CCMAES::printFinal() const
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("--------------------------------------------------------------------\n");
    printf("[Korali] CMA-ES Finished\n");
    printf("[Korali] Optimum (%s) found: %e\n", _objective.c_str(), bestEver);
    printf("[Korali] Optimum (%s) found at:\n", _objective.c_str());
    for (size_t i = 0; i < _k->_problem->N; i++) printf("         %s = %+6.3e\n", _k->_problem->_parameters[i]->_name.c_str(), rgxbestever[i]);
    printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
    printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
    printf("[Korali] Stopping Criterium: %s\n", _terminationReason);
    printf("[Korali] Total Elapsed Time: %fs\n", std::chrono::duration<double>(endTime-startTime).count());
 }
}
