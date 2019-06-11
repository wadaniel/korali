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
 _samplePopulation =  (double*) calloc (sizeof(double), _k->_problem->N*_s);
 
 rgpc           = (double*) calloc (sizeof(double), _k->_problem->N);
 rgps           = (double*) calloc (sizeof(double), _k->_problem->N);
 rgdTmp         = (double*) calloc (sizeof(double), _k->_problem->N);
 rgBDz          = (double*) calloc (sizeof(double), _k->_problem->N);
 rgxmean        = (double*) calloc (sizeof(double), _k->_problem->N);
 rgxold         = (double*) calloc (sizeof(double), _k->_problem->N);
 rgxbestever    = (double*) calloc (sizeof(double), _k->_problem->N);
 axisD          = (double*) calloc (sizeof(double), _k->_problem->N);
 axisDtmp       = (double*) calloc (sizeof(double), _k->_problem->N);
 curBestVector  = (double*) calloc (sizeof(double), _k->_problem->N);

 index          = (size_t*) calloc (sizeof(size_t), _s);
 histFuncValues = (double*) calloc (sizeof(double), _termCondMaxGenerations+1);

 C    = (double**) calloc (sizeof(double*), _k->_problem->N);
 Ctmp = (double**) calloc (sizeof(double*), _k->_problem->N);
 B    = (double**) calloc (sizeof(double*), _k->_problem->N);
 Btmp = (double**) calloc (sizeof(double*), _k->_problem->N);
 for (size_t i = 0; i < _k->_problem->N; i++) C[i] = (double*) calloc (sizeof(double), _k->_problem->N);
 for (size_t i = 0; i < _k->_problem->N; i++) Ctmp[i] = (double*) calloc (sizeof(double), _k->_problem->N);
 for (size_t i = 0; i < _k->_problem->N; i++) B[i] = (double*) calloc (sizeof(double), _k->_problem->N);
 for (size_t i = 0; i < _k->_problem->N; i++) Btmp[i] = (double*) calloc (sizeof(double), _k->_problem->N);

 _initializedSample = (bool*) calloc (sizeof(bool), _s);
 _fitnessVector = (double*) calloc (sizeof(double), _s);

 // Init Generation
 _currentGeneration = 0;
 
 // Initializing Gaussian Generator
 _gaussianGenerator = new Variable::Gaussian(0.0, 1.0, _k->_seed++);

 _chiN = sqrt((double) _k->_problem->N) * (1. - 1./(4.*_k->_problem->N) + 1./(21.*_k->_problem->N*_k->_problem->N));
 
 // Initailizing Mu
 _muWeights = (double *) calloc (sizeof(double), _mu);
 
 // Setting Mu & Dependent Variables
 initInternals();

 // Setting eigensystem evaluation Frequency
 if( _covarianceEigenEvalFreq < 0.0) _covarianceEigenEvalFreq = 1.0/_covarianceMatrixLearningRate/((double)_k->_problem->N)/10.0;

 flgEigensysIsUptodate = true;

 countevals = 0;
 countinfeasible = 0;
 resampled = 0;
 bestEver = -std::numeric_limits<double>::max();

 minEW = doubleRangeMin(axisD, _k->_problem->N); minEW = minEW * minEW;
 maxEW = doubleRangeMax(axisD, _k->_problem->N); maxEW = maxEW * maxEW;

 maxdiagC=C[0][0]; for(size_t i = 1; i<_k->_problem->N; i++) if(maxdiagC<C[i][i]) maxdiagC=C[i][i];
 mindiagC=C[0][0]; for(size_t i = 1; i<_k->_problem->N; i++) if(mindiagC>C[i][i]) mindiagC=C[i][i];

 psL2 = 0.0;

 /* set rgxmean */
 for (size_t i = 0; i < _k->_problem->N; ++i)
 {
   if(_initialMeans[i] < _lowerBounds[i] || _initialMeans[i] > _upperBounds[i])
    fprintf(stderr,"[Korali] Warning: Initial Value (%.4f) for \'%s\' is out of bounds (%.4f-%.4f).\n", 
            _initialMeans[i],
            _k->_problem->_variables[i]->_name.c_str(), 
            _lowerBounds[i],
            _upperBounds[i]);
   rgxmean[i] = rgxold[i] = _initialMeans[i];
 }

 Z   = (double**) malloc (sizeof(double*) * _s);
 BDZ = (double**) malloc (sizeof(double*) * _s);
 for (size_t i = 0; i < _s; i++) Z[i]   = (double*) calloc (sizeof(double), _k->_problem->N);
 for (size_t i = 0; i < _s; i++) BDZ[i] = (double*) calloc (sizeof(double), _k->_problem->N);

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

 js["Sample Count"]             = _s;
 js["Sigma Cumulation Factor"] = _sigmaCumulationFactor;
 js["Damp Factor"]             = _dampFactor;
 js["Objective"]               = _objective;
 js["Max Resamplings"]         = _maxResamplings;
 js["Sigma Bounded"]           = _isSigmaBounded;
 
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
 js["Termination Criteria"]["Fitness"]["Value"]                   = _termCondFitness;
 js["Termination Criteria"]["Fitness"]["Active"]                  = _isTermCondFitness;
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
 for (size_t i = 0; i < _s; i++) js["State"]["FunctionValues"] += _fitnessVector[i];
 
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["CurrentMeanVector"]      += rgxmean[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["PreviousMeanVector"]     += rgxold[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["BestEverVector"]         += rgxbestever[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["CurrentBestVector"]      += curBestVector[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["AxisLengths"]            += axisD[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["EvolutionPath"]          += rgpc[i];
 for (size_t i = 0; i < _k->_problem->N; i++) js["State"]["ConjugateEvolutionPath"] += rgps[i];

 for (size_t i = 0; i < _s; i++) for (size_t j = 0; j < _k->_problem->N; j++) js["State"]["Samples"][i][j] = _samplePopulation[i*_k->_problem->N + j];
 for (size_t i = 0; i < _k->_problem->N; i++) for (size_t j = 0; j <= i; j++) { js["State"]["CovarianceMatrix"][i][j] = C[i][j]; js["State"]["CovarianceMatrix"][j][i] = C[i][j]; }
 for (size_t i = 0; i < _k->_problem->N; i++) for (size_t j = 0; j <= i; j++) { js["State"]["EigenMatrix"][i][j] = B[i][j]; js["State"]["EigenMatrix"][j][i] = B[i][j]; }

 return js;
}


void CMAES::setConfiguration(nlohmann::json& js)
{
 this->Korali::Solver::Base::setConfiguration(js);

 _s                             = consume(js, { "Sample Count" }, KORALI_NUMBER);
 _sigmaCumulationFactorIn       = consume(js, { "Sigma Cumulation Factor" }, KORALI_NUMBER, std::to_string(-1));
 _dampFactorIn                  = consume(js, { "Damp Factor" }, KORALI_NUMBER, std::to_string(-1));
 _objective                     = consume(js, { "Objective" }, KORALI_STRING, "Maximize");
 _maxResamplings                = consume(js, { "Max Resamplings" }, KORALI_NUMBER, std::to_string(1e6));
 _isSigmaBounded                = consume(js, { "Sigma Bounded" }, KORALI_BOOLEAN, "false");

 _fitnessSign = 0;
 if(_objective == "Maximize") _fitnessSign = 1;
 if(_objective == "Minimize") _fitnessSign = -1;
 if(_fitnessSign == 0)  { fprintf( stderr, "[Korali] Error: Invalid setting for Objective: %s\n", _objective.c_str()); exit(-1); }

 _mu                            = consume(js, { "Mu", "Value" }, KORALI_NUMBER, std::to_string(ceil(_s / 2)));
 _muType                        = consume(js, { "Mu", "Type" }, KORALI_STRING, "Logarithmic");
 _muCovarianceIn                = consume(js, { "Mu", "Covariance" }, KORALI_NUMBER, std::to_string(-1));

 // Initializing Mu Weights
 _muWeights = (double *) calloc (sizeof(double), _mu);
 if (_muType == "Linear")       for (size_t i = 0; i < _mu; i++)  _muWeights[i] = _mu - i;
 else if (_muType == "Equal")        for (size_t i = 0; i < _mu; i++)  _muWeights[i] = 1;
 else if (_muType == "Logarithmic")  for (size_t i = 0; i < _mu; i++)  _muWeights[i] = log(_mu+1.)-log(i+1.);
 else  { fprintf( stderr, "[Korali] Error: Invalid setting of Mu Type (%s) (Linear, Equal or Logarithmic accepted).", _muType.c_str()); exit(-1); }
 
 if(_mu < 1 || _mu > _s || (( _mu == _s )  && _muType.compare("Linear") ))
   { fprintf( stderr, "[Korali] Error: Invalid setting of Mu (%lu) and/or Lambda (%lu)\n", _mu, _s); exit(-1); }
 
 _covarianceEigenEvalFreq      = consume(js, { "Covariance Matrix", "Eigenvalue Evaluation Frequency" }, KORALI_NUMBER, std::to_string(0));
 _cumulativeCovarianceIn       = consume(js, { "Covariance Matrix", "Cumulative Covariance" }, KORALI_NUMBER, std::to_string(-1));
 _covMatrixLearningRateIn      = consume(js, { "Covariance Matrix", "Learning Rate" }, KORALI_NUMBER, std::to_string(-1));
 _isdiag                       = consume(js, { "Covariance Matrix", "Is Diagonal" }, KORALI_BOOLEAN, "false");

 // Setting variable information

 auto vec = consume(js, { "Initial Means" }, KORALI_ARRAY);
 for (size_t i = 0; i < vec.size(); i++) _initialMeans.push_back(vec[i]);

 vec = consume(js, { "Initial Standard Deviations" }, KORALI_ARRAY);
 for (size_t i = 0; i < vec.size(); i++) _initialStdDevs.push_back(vec[i]);

 vec = consume(js, { "Minimum Standard Deviation Changes" }, KORALI_ARRAY);
 for (size_t i = 0; i < vec.size(); i++) _minStdDevChanges.push_back(vec[i]);

 vec = consume(js, { "Lower Bounds" }, KORALI_ARRAY);
 for (size_t i = 0; i < vec.size(); i++) _lowerBounds.push_back(vec[i]);

 vec = consume(js, { "Upper Bounds" }, KORALI_ARRAY);
 for (size_t i = 0; i < vec.size(); i++) _upperBounds.push_back(vec[i]);

 // Setting termination criteria

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
 currentFunctionValue  = js["State"]["CurrentBestFunctionValue"];
 prevBest              = js["State"]["PreviousBestFunctionValue"];
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
 for (size_t i = 0; i < _k->_problem->N; i++) axisD[i]         = js["State"]["AxisLengths"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) rgpc[i]          = js["State"]["EvolutionPath"][i];
 for (size_t i = 0; i < _k->_problem->N; i++) curBestVector[i] = js["State"]["CurrentBestVector"][i];
 for (size_t i = 0; i < _s; i++) index[i]              = js["State"]["Index"][i];
 for (size_t i = 0; i < _s; i++) _fitnessVector[i]     = js["State"]["FunctionValues"][i];
 for (size_t i = 0; i < _s; i++) for (size_t j = 0; j < _k->_problem->N; j++) _samplePopulation[i*_k->_problem->N + j] = js["State"]["Samples"][i][j];
 for (size_t i = 0; i < _k->_problem->N; i++) for (size_t j = 0; j < _k->_problem->N; j++) C[i][j] = js["State"]["CovarianceMatrix"][i][j];
 for (size_t i = 0; i < _k->_problem->N; i++) for (size_t j = 0; j < _k->_problem->N; j++) B[i][j] = js["State"]["EigenMatrix"][i][j];

}


void CMAES::initInternals()
{
 // Setting variable information

 if (_lowerBounds.size() == 0) for(size_t i = 0; i < _k->_problem->N; i++) _lowerBounds.push_back(-INFINITY);
 else if (_lowerBounds.size() != _k->_problem->N) { fprintf( stderr, "[Korali] Error: Incorrect number of lower bounds defined. Expected: %lu, Provided: %lu\n.", _k->_problem->N, _lowerBounds.size()); exit(-1); }

 if (_upperBounds.size() == 0) for(size_t i = 0; i < _k->_problem->N; i++) _upperBounds.push_back(+INFINITY);
 else if (_upperBounds.size() != _k->_problem->N) { fprintf( stderr, "[Korali] Error: Incorrect number of upper bounds defined. Expected: %lu, Provided: %lu\n.", _k->_problem->N, _upperBounds.size()); exit(-1); }

 for(size_t i = 0; i < _k->_problem->N; i++) if (_upperBounds[i] <= _lowerBounds[i])
 { fprintf( stderr, "[Korali] Error: Invalid Lower (%f) - Upper (%f) bounds range defined for variable %d.\n", _lowerBounds[i], _upperBounds[i], i); exit(-1); }

 if (_initialMeans.size() == 0)
 for(size_t i = 0; i < _k->_problem->N; i++)
 {
  if( isinf(_upperBounds[i]) || isinf(_lowerBounds[i]) )
  {
    fprintf(stderr, "[Korali] Error: Either or both lower or upper bounds of variable %lu is infinite, you need to define its Initial Mean.", i);
    exit(-1);
  }
  _initialMeans.push_back(0.5*(_lowerBounds[i]+_upperBounds[i]));
 }
 else if (_upperBounds.size() != _k->_problem->N) { fprintf( stderr, "[Korali] Error: Incorrect number of Initial Means defined. Expected: %lu, Provided: %lu\n.", _k->_problem->N, _initialMeans.size()); exit(-1); }

 if (_initialStdDevs.size() == 0)
 for(size_t i = 0; i < _k->_problem->N; i++)
 {
  if( isinf(_upperBounds[i]) || isinf(_lowerBounds[i]) )
  {
    fprintf(stderr, "[Korali] Error: Either or both lower or upper bounds of variable %lu is infinite, you need to define its Initial Standard Deviation.", i);
    exit(-1);
  }
  _initialStdDevs.push_back(0.3*(_upperBounds[i]-_lowerBounds[i]));
 }
 else if (_initialStdDevs.size() != _k->_problem->N) { fprintf( stderr, "[Korali] Error: Incorrect number of Initial Standard Deviations defined. Expected: %lu, Provided: %lu\n.", _k->_problem->N, _initialStdDevs.size()); exit(-1); }

 for(size_t i = 0; i < _k->_problem->N; i++)
  if (_initialStdDevs[i] < 0.0) { fprintf(stderr, "[Korali] Error: Initial StdDev for variable %d is less or equal 0.\n", i);  exit(-1); }

 if (_minStdDevChanges.size() == 0) for(size_t i = 0; i < _k->_problem->N; i++) _minStdDevChanges.push_back(0.0);
 else if (_minStdDevChanges.size() != _k->_problem->N) { fprintf( stderr, "[Korali] Error: Incorrect number of Minimum Standard Deviation Changes defined. Expected: %lu, Provided: %lu\n.", _k->_problem->N, _minStdDevChanges.size()); exit(-1); }

 // Initializing Mu Weights

 if      (_muType == "Linear")      for (size_t i = 0; i < _mu; i++) _muWeights[i] = _mu - i;
 else if (_muType == "Equal")       for (size_t i = 0; i < _mu; i++) _muWeights[i] = 1;
 else if (_muType == "Logarithmic") for (size_t i = 0; i < _mu; i++) _muWeights[i] = log(std::max( (double)_mu, 0.5*_s)+0.5)-log(i+1.);
 else  { fprintf( stderr, "[Korali] Error: Invalid setting of Mu Type (%s) (Linear, Equal or Logarithmic accepted).", _muType.c_str()); exit(-1); }

 // Normalize weights vector and set mueff
 double s1 = 0.0;
 double s2 = 0.0;

 for (size_t  i = 0; i < _mu; i++)
 {
  s1 += _muWeights[i];
  s2 += _muWeights[i]*_muWeights[i];
 }
 _muEffective = s1*s1/s2;

 for (size_t i = 0; i < _mu; i++) _muWeights[i] /= s1;

 // Setting Mu Covariance
 if (_muCovarianceIn < 1) _muCovariance = _muEffective;
 else                     _muCovariance = _muCovarianceIn;
 
 // Setting Cumulative Covariancea
 if( (_cumulativeCovarianceIn <= 0) || (_cumulativeCovarianceIn > 1) ) 
     _cumulativeCovariance = (4.0 + _muEffective/(1.0*_k->_problem->N)) / (_k->_problem->N+4.0 + 2.0*_muEffective/(1.0*_k->_problem->N));
 else _cumulativeCovariance = _cumulativeCovarianceIn;

 // Setting Covariance Matrix Learning Rate
 double l1 = 2. / ((_k->_problem->N+1.4142)*(_k->_problem->N+1.4142));          
 double l2 = (2.*_muEffective-1.) / ((_k->_problem->N+2.)*(_k->_problem->N+2.)+_muEffective);
 l2 = (l2 > 1) ? 1 : l2;                                                        
 l2 = (1./_muCovariance) * l1 + (1.-1./_muCovariance) * l2;                     
 
 _covarianceMatrixLearningRate = _covMatrixLearningRateIn;
 if (_covarianceMatrixLearningRate < 0 || _covarianceMatrixLearningRate > 1)  _covarianceMatrixLearningRate = l2;

  // Setting Sigma Cumulation Factor
 _sigmaCumulationFactor = _sigmaCumulationFactorIn;
 if (_sigmaCumulationFactor <= 0 || _sigmaCumulationFactor >= 1) _sigmaCumulationFactor = (_muEffective + 2.) / (_k->_problem->N + _muEffective + 3.0);

 // Setting Damping Factor
 _dampFactor = _dampFactorIn;
 if (_dampFactor <= 0.0) 
     _dampFactor = 1.0 * (1.0 + 2*std::max(0.0, sqrt((_muEffective-1.0)/(_k->_problem->N+1.0)) - 1))  /* basic factor */
        // * std::max(0.3, 1. - (double)_k->_problem->N / (1e-6+std::min(_termCondMaxGenerations, _termCondMaxFitnessEvaluations/_via_s))) /* modification for short runs */
        + _sigmaCumulationFactor; /* minor increment */
 
 // Setting Sigma
 _trace = 0.0;
 for (size_t i = 0; i < _k->_problem->N; ++i) _trace += _initialStdDevs[i]*_initialStdDevs[i];
 sigma = sqrt(_trace/_k->_problem->N); /* _muEffective/(0.2*_muEffective+sqrt(_k->_problem->N)) * sqrt(_trace/_k->_problem->N); */

 // Setting B, C and axisD
 for (size_t i = 0; i < _k->_problem->N; ++i)
 {
  B[i][i] = 1.0;
  C[i][i] = axisD[i] = _initialStdDevs[i] * sqrt(_k->_problem->N / _trace);
  C[i][i] *= C[i][i];
 }

 minEW = doubleRangeMin(axisD, _k->_problem->N); minEW = minEW * minEW;
 maxEW = doubleRangeMax(axisD, _k->_problem->N); maxEW = maxEW * maxEW;

 maxdiagC = C[0][0]; for(size_t i = 1; i < _k->_problem->N; i++) if( maxdiagC < C[i][i]) maxdiagC = C[i][i];
 mindiagC = C[0][0]; for(size_t i = 1; i < _k->_problem->N; i++) if( mindiagC > C[i][i]) mindiagC = C[i][i];

}


/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/


void CMAES::run()
{
 if (_k->_verbosity >= KORALI_MINIMAL) { 
   printf("[Korali] Starting CMA-ES (Objective: %s).\n", _objective.c_str());
   printf("--------------------------------------------------------------------\n");
 }

 startTime = std::chrono::system_clock::now();
 _k->saveState(_currentGeneration);

 while(!checkTermination())
 {
   t0 = std::chrono::system_clock::now();
   prepareGeneration();
   evaluateSamples();
   updateDistribution(_fitnessVector);
   _currentGeneration++;
   t1 = std::chrono::system_clock::now();
   printGeneration();
   _k->saveState(_currentGeneration);
 }

 endTime = std::chrono::system_clock::now();
 printFinal();

}


void CMAES::evaluateSamples() 
{
  while (_finishedSamples < _s)
  {
    for (size_t i = 0; i < _s; i++) if (_initializedSample[i] == false)
    {
      _initializedSample[i] = true;
      _k->_conduit->evaluateSample(_samplePopulation, i); countevals++;
    }
    _k->_conduit->checkProgress();
  }
}


void CMAES::processSample(size_t sampleId, double fitness)
{
 _fitnessVector[sampleId] = _fitnessSign*fitness;
 _finishedSamples++;
}


bool CMAES::isFeasible(size_t sampleIdx) const
{
 for (size_t d = 0; d < _k->_problem->N; ++d)
  if (_samplePopulation[ sampleIdx*_k->_problem->N+d ] < _lowerBounds[d] || _samplePopulation[ sampleIdx*_k->_problem->N+d ] > _upperBounds[d]) return false;
 return true;
}


void CMAES::prepareGeneration()
{

 /* calculate eigensystem */
 updateEigensystem(C);

 for (size_t i = 0; i < _s; ++i) 
 {
     size_t initial_infeasible = countinfeasible;
     sampleSingle(i);
     while( isFeasible( i ) == false ) 
     { 
       countinfeasible++; 
       sampleSingle(i); 
       if ( (countinfeasible - initial_infeasible) > _maxResamplings ) 
       { 
        if(_k->_verbosity >= KORALI_DETAILED) printf("[Korali] Warning: exiting resampling loop (param %zu) , max resamplings (%zu) reached.\n", i, _maxResamplings);
        exit(-1);
       }
     } 
 }


 _finishedSamples = 0;
 for (size_t i = 0; i < _s; i++) _initializedSample[i] = false;
}


void CMAES::sampleSingle(size_t sampleIdx) 
{

  /* generate scaled random vector (D * z) */
  for (size_t d = 0; d < _k->_problem->N; ++d)
  {
   Z[sampleIdx][d] = _gaussianGenerator->getRandomNumber();
   if (_isdiag) { 
     BDZ[sampleIdx][d] = axisD[d] * Z[sampleIdx][d];
     _samplePopulation[sampleIdx * _k->_problem->N + d] = rgxmean[d] + sigma * BDZ[sampleIdx][d]; 
   }
   else rgdTmp[d] = axisD[d] * Z[sampleIdx][d];
  }

  if (!_isdiag)
   for (size_t d = 0; d < _k->_problem->N; ++d) {
    BDZ[sampleIdx][d] = 0.0;
    for (size_t e = 0; e < _k->_problem->N; ++e) BDZ[sampleIdx][d] += B[d][e] * rgdTmp[e];
    _samplePopulation[sampleIdx * _k->_problem->N + d] = rgxmean[d] + sigma * BDZ[sampleIdx][d];
  }
}


void CMAES::updateDistribution(const double *fitnessVector)
{

 /* Generate index */
 sort_index(fitnessVector, index, _s);
 
 /* update function value history */
 prevFunctionValue = currentFunctionValue;

 /* update current best */
 currentFunctionValue = fitnessVector[index[0]];
 for (size_t d = 0; d < _k->_problem->N; ++d) curBestVector[d] = _samplePopulation[index[0]*_k->_problem->N+d];
 
 /* update xbestever */
 //TODO: what if we minimize
 if ( currentFunctionValue > bestEver )
 {
  prevBest = bestEver;
  bestEver = currentFunctionValue;
  for (size_t d = 0; d < _k->_problem->N; ++d) rgxbestever[d]   = curBestVector[d];
 }

 histFuncValues[_currentGeneration] = bestEver;

 /* calculate rgxmean and rgBDz */
 for (size_t d = 0; d < _k->_problem->N; ++d) {
   rgxold[d] = rgxmean[d];
   rgxmean[d] = 0.;
   for (size_t i = 0; i < _mu; ++i)
     rgxmean[d] += _muWeights[i] * _samplePopulation[index[i]*_k->_problem->N + d];
   
   rgBDz[d] = (rgxmean[d] - rgxold[d])/sigma;
 }

 /* calculate z := D^(-1) * B^(T) * rgBDz into rgdTmp */
 for (size_t d = 0; d < _k->_problem->N; ++d) {
  double sum = 0.0;
  if (_isdiag) sum = rgBDz[d];
  else for (size_t e = 0; e < _k->_problem->N; ++e) sum += B[e][d] * rgBDz[e]; /* B^(T) * rgBDz ( iterating B[e][d] = B^(T) ) */
  
  rgdTmp[d] = sum / axisD[d]; /* D^(-1) * B^(T) * rgBDz */
 }

 psL2 = 0.0;

 /* cumulation for sigma (ps) using B*z */
 for (size_t d = 0; d < _k->_problem->N; ++d) {
    double sum = 0.0;
    if (_isdiag) sum = rgdTmp[d];
    else for (size_t e = 0; e < _k->_problem->N; ++e) sum += B[d][e] * rgdTmp[e];

    rgps[d] = (1. - _sigmaCumulationFactor) * rgps[d] + sqrt(_sigmaCumulationFactor * (2. - _sigmaCumulationFactor) * _muEffective) * sum;
 
    /* calculate norm(ps)^2 */
    psL2 += rgps[d] * rgps[d];
 }

 int hsig = sqrt(psL2) / sqrt(1. - pow(1.-_sigmaCumulationFactor, 2.0*(1.0+_currentGeneration))) / _chiN  < 1.4 + 2./(_k->_problem->N+1);

 /* cumulation for covariance matrix (pc) using B*D*z~_k->_problem->N(0,C) */
 for (size_t d = 0; d < _k->_problem->N; ++d)
   rgpc[d] = (1. - _cumulativeCovariance) * rgpc[d] + hsig * sqrt( _cumulativeCovariance * (2. - _cumulativeCovariance) * _muEffective ) * rgBDz[d];

 /* update of C  */
 adaptC(hsig);

 // update sigma & viability boundaries
 //sigma *= exp(((sqrt(psL2)/_chiN)-1.)*_sigmaCumulationFactor/_dampFactor) (orig, alternative)
 sigma *= exp(std::min(1.0, ((sqrt(psL2)/_chiN)-1.)*_sigmaCumulationFactor/_dampFactor));
 
 /* upper bound for sigma */
 double sigma_upper = sqrt(_trace/_k->_problem->N);
 if( _isSigmaBounded && (sigma > sigma_upper) ) {
     fprintf(stderr, "[Korali] Warning: sigma bounded by %f, increase Initial Std of vairables.\n", sigma_upper);
     sigma = sigma_upper;
 }
 
 
 /* numerical error management */
 
 //treat maximal standard deviations
 //TODO

 //treat minimal standard deviations
 for (size_t d = 0; d < _k->_problem->N; ++d) if (sigma * sqrt(C[d][d]) < _minStdDevChanges[d])
 {
   sigma = (_minStdDevChanges[d])/sqrt(C[d][d]) * exp(0.05+_sigmaCumulationFactor/_dampFactor);
   if (_k->_verbosity >= KORALI_DETAILED) fprintf(stderr, "[Korali] Warning: sigma increased due to minimal standard deviation.\n");
 }

 //too low coordinate axis deviations
 //TODO
 
 //treat numerical precision provblems
 //TODO

 /* Test if function values are identical, escape flat fitness */
 if (currentFunctionValue == _fitnessVector[index[(int)_mu]]) {
   sigma *= exp(0.2+_sigmaCumulationFactor/_dampFactor);
   if (_k->_verbosity >= KORALI_DETAILED) {
     fprintf(stderr, "[Korali] Warning: sigma increased due to equal function values.\n");
   }
 }

 /*
 size_t horizon = 10 + ceil(3*10*_k->_problem->N/_s);
 double min = std::numeric_limits<double>::max();
 double max = -std::numeric_limits<double>::max();
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
  //double ccov1  = std::min(_covarianceMatrixLearningRate * (1./_muCovariance) * (_isdiag ? (_k->_problem->N+1.5) / 3. : 1.), 1.); (orig, alternative)
  //double ccovmu = std::min(_covarianceMatrixLearningRate * (1-1./_muCovariance) * (_isdiag ? (_k->_problem->N+1.5) / 3. : 1.), 1.-ccov1); (orig, alternative)
  double ccov1  = 2.0 / (std::pow(_k->_problem->N+1.3,2)+_muEffective);
  double ccovmu = std::min(1.0-ccov1,  2.0 * (_muEffective-2+1/_muEffective) / (std::pow(_k->_problem->N+2.0,2)+_muEffective));
  double sigmasquare = sigma * sigma;

  flgEigensysIsUptodate = false;

  /* update covariance matrix */
  for (size_t d = 0; d < _k->_problem->N; ++d)
   for (size_t e = _isdiag ? d : 0; e <= d; ++e) {
     C[d][e] = (1 - ccov1 - ccovmu) * C[d][e] + ccov1 * (rgpc[d] * rgpc[e] + (1-hsig)*ccov1*_cumulativeCovariance*(2.-_cumulativeCovariance) * C[d][e]);
     for (size_t k = 0; k < _mu; ++k) C[d][e] += ccovmu * _muWeights[k] * (_samplePopulation[index[k]*_k->_problem->N + d] - rgxold[d]) * (_samplePopulation[index[k]*_k->_problem->N + e] - rgxold[e]) / sigmasquare;
     if (e < d) C[e][d] = C[d][e];
   }

  /* update maximal and minimal diagonal value */
  maxdiagC = mindiagC = C[0][0];
  for (size_t d = 1; d < _k->_problem->N; ++d) {
  if (maxdiagC < C[d][d]) maxdiagC = C[d][d];
  else if (mindiagC > C[d][d])  mindiagC = C[d][d];
  }
 } /* if ccov... */
}


bool CMAES::checkTermination()
{
 double fac;

 bool terminate = false;

 if ( _isTermCondFitness  && (_currentGeneration > 1) && (bestEver >= _termCondFitness) )
 {
  terminate = true;
  sprintf(_terminationReason, "Fitness Value (%+6.3e) > (%+6.3e)",  bestEver, _termCondFitness);
 }

 double range = fabs(currentFunctionValue - prevFunctionValue);
 if ( _isTermCondFitnessDiffThreshold && (_currentGeneration > 1) && (range <= _termCondFitnessDiffThreshold) )
 {
  terminate = true;
  sprintf(_terminationReason, "Function value differences (%+6.3e) < (%+6.3e)",  range, _termCondFitnessDiffThreshold);
 }

 size_t cTemp = 0;
 for(size_t i=0; i<_k->_problem->N; ++i) {
  cTemp += (sigma * sqrt(C[i][i]) < _termCondMinDeltaX) ? 1 : 0;
  cTemp += (sigma * rgpc[i] < _termCondMinDeltaX) ? 1 : 0;
 }

 if ( _isTermCondMinDeltaX && (cTemp == 2*_k->_problem->N) ) {
  terminate = true;
  sprintf(_terminationReason, "Object variable changes < %+6.3e", _termCondMinDeltaX);
 }

 for(size_t i=0; i<_k->_problem->N; ++i)
  if ( _isTermCondTolUpXFactor && (sigma * sqrt(C[i][i]) > _termCondTolUpXFactor * _initialStdDevs[i]) )
  {
    terminate = true;
    sprintf(_terminationReason, "Standard deviation increased by more than %7.2e, larger initial standard deviation recommended \n", _termCondTolUpXFactor);
    break;
  }

 if ( _isTermCondCovCond && (maxEW >= minEW * _termCondCovCond) )
 {
   terminate = true;
   sprintf(_terminationReason, "Maximal condition number %7.2e reached. maxEW=%7.2e, minEig=%7.2e, maxdiagC=%7.2e, mindiagC=%7.2e\n",
                                _termCondCovCond, maxEW, minEW, maxdiagC, mindiagC);
 }

  size_t iAchse = 0;
  size_t iKoo = 0;
  if (!_isdiag ) // TODO: no effect axis add to _isTermCond..
  {
    for (iAchse = 0; iAchse < _k->_problem->N; ++iAchse)
    {
    fac = 0.1 * sigma * axisD[iAchse];
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
 } /* if _isdiag */

 /* Component of rgxmean is not changed anymore */
 for (iKoo = 0; iKoo < _k->_problem->N; ++iKoo)
 {
  if (rgxmean[iKoo] == rgxmean[iKoo] + 0.2*sigma*sqrt(C[iKoo][iKoo]) ) //TODO: standard dev add to _isTermCond..
  {
   /* C[iKoo][iKoo] *= (1 + _covarianceMatrixLearningRate); */
   /* flg = 1; */
   terminate = true;
   sprintf(_terminationReason, "Standard deviation 0.2*%7.2e in coordinate %ld without effect.", sigma*sqrt(C[iKoo][iKoo]), iKoo);
   break;
  }

 } /* for iKoo */

 if( _isTermCondMaxFitnessEvaluations && (countevals >= _termCondMaxFitnessEvaluations) )
 {
  terminate = true;
  sprintf(_terminationReason, "Conducted %lu function evaluations >= (%lu).", countevals, _termCondMaxFitnessEvaluations);
 }

 if( _isTermCondMaxGenerations && (_currentGeneration >= _termCondMaxGenerations) )
 {
  terminate = true;
  sprintf(_terminationReason, "Maximum number of Generations reached (%lu).", _termCondMaxGenerations);
 }

 return terminate;
}


void CMAES::updateEigensystem(double **M, int flgforce)
{
 if(flgforce == 0 && flgEigensysIsUptodate) return;
 /* if(_currentGeneration % _covarianceEigenEvalFreq == 0) return; */

 eigen(_k->_problem->N, M, axisDtmp, Btmp);
  
 /* find largest and smallest eigenvalue, they are supposed to be sorted anyway */
 double minEWtmp = doubleRangeMin(axisDtmp, _k->_problem->N);
 double maxEWtmp = doubleRangeMax(axisDtmp, _k->_problem->N);

 if (minEWtmp <= 0.0)
  { printf("[Korali] Warning: Min Eigenvalue smaller or equal 0.0 (%+6.3e) after Eigen decomp (no update possible).\n", minEWtmp ); return; }

 /* write back */
 for (size_t d = 0; d < _k->_problem->N; ++d) axisD[d] = sqrt(axisDtmp[d]);
 for (size_t d = 0; d < _k->_problem->N; ++d) std::memcpy( B[d], Btmp[d], sizeof(double) * _k->_problem->N );
 for (size_t d = 0; d < _k->_problem->N; ++d) std::memcpy( C[d], M[d], sizeof(double) * _k->_problem->N );

 minEW = minEWtmp;
 maxEW = maxEWtmp;

 flgEigensysIsUptodate = true;
}


/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/


void CMAES::eigen(size_t size, double **M, double *diag, double **Q) const
{
 double* data = (double*) malloc (sizeof(double) * size * size);

 for (size_t i = 0; i <  size; i++)
 for (size_t j = 0; j <= i; j++)
 {
  data[i*size + j] = M[i][j];
  data[j*size + i] = M[i][j];
 }

 gsl_matrix_view m = gsl_matrix_view_array (data, size, size);
 gsl_vector *eval  = gsl_vector_alloc (size);
 gsl_matrix *evec  = gsl_matrix_alloc (size, size);
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
  std::sort( index, index+n, [fitnessVector](size_t i1, size_t i2) {return fitnessVector[i1] > fitnessVector[i2];} ); //descending (TODO: ok with minimize?)

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

 if (_k->_verbosity >= KORALI_MINIMAL)
   printf("[Korali] Generation %ld - Duration: %fs (Total Elapsed Time: %fs)\n", _currentGeneration, std::chrono::duration<double>(t1-t0).count(), std::chrono::duration<double>(t1-startTime).count());
 
 if (_k->_verbosity >= KORALI_NORMAL)
 {
  printf("[Korali] Sigma:                        %+6.3e\n", sigma);
  printf("[Korali] Current Function Value: Max = %+6.3e - Best = %+6.3e\n", currentFunctionValue, bestEver);
  printf("[Korali] Diagonal Covariance:    Min = %+6.3e -  Max = %+6.3e\n", mindiagC, maxdiagC);
  printf("[Korali] Covariance Eigenvalues: Min = %+6.3e -  Max = %+6.3e\n", minEW, maxEW);
 }

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _k->_problem->N; d++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_problem->_variables[d]->_name.c_str(), rgxmean[d], rgxbestever[d]);

  printf("[Korali] Covariance Matrix:\n");
  for (size_t d = 0; d < _k->_problem->N; d++)
  {
   for (size_t e = 0; e <= d; e++) printf("   %+6.3e  ",C[d][e]);
   printf("\n");
  }

  printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
  printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
 } 
 
 if (_k->_verbosity >= KORALI_NORMAL)
   printf("--------------------------------------------------------------------\n");


}


void CMAES::printFinal() const
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("[Korali] CMA-ES Finished\n");
    printf("[Korali] Optimum (%s) found: %e\n", _objective.c_str(), bestEver);
    printf("[Korali] Optimum (%s) found at:\n", _objective.c_str());
    for (size_t d = 0; d < _k->_problem->N; ++d) printf("         %s = %+6.3e\n", _k->_problem->_variables[d]->_name.c_str(), rgxbestever[d]);
    printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
    printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
    printf("[Korali] Stopping Criterium: %s\n", _terminationReason);
    printf("[Korali] Total Elapsed Time: %fs\n", std::chrono::duration<double>(endTime-startTime).count());
    printf("--------------------------------------------------------------------\n");
 }
}
