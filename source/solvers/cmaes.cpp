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

CMAES::CMAES(nlohmann::json& js, std::string name)
{
 _name = name;
 
 setConfiguration(js);

 size_t s_max, mu_max;
 if (_name == "CCMA-ES")
 {
   s_max  = std::max(_s, _via_s);
   mu_max = std::max(_mu, _via_mu);
 }
 else
 {
   s_max  = _s;
   mu_max = _mu;
 }

 // Allocating Memory
 _samplePopulation =  (double*) calloc (sizeof(double), _k->N*s_max);

 rgpc           = (double*) calloc (sizeof(double), _k->N);
 rgps           = (double*) calloc (sizeof(double), _k->N);
 rgdTmp         = (double*) calloc (sizeof(double), _k->N);
 rgBDz          = (double*) calloc (sizeof(double), _k->N);
 rgxmean        = (double*) calloc (sizeof(double), _k->N);
 rgxold         = (double*) calloc (sizeof(double), _k->N);
 rgxbestever    = (double*) calloc (sizeof(double), _k->N);
 axisD          = (double*) calloc (sizeof(double), _k->N);
 axisDtmp       = (double*) calloc (sizeof(double), _k->N);
 curBestVector  = (double*) calloc (sizeof(double), _k->N);

 index          = (size_t*) calloc (sizeof(size_t), s_max);
 histFuncValues = (double*) calloc (sizeof(double), _termCondMaxGenerations+1);

 C    = (double**) calloc (sizeof(double*), _k->N);
 Ctmp = (double**) calloc (sizeof(double*), _k->N);
 B    = (double**) calloc (sizeof(double*), _k->N);
 Btmp = (double**) calloc (sizeof(double*), _k->N);
 for (size_t i = 0; i < _k->N; i++) C[i] = (double*) calloc (sizeof(double), _k->N);
 for (size_t i = 0; i < _k->N; i++) Ctmp[i] = (double*) calloc (sizeof(double), _k->N);
 for (size_t i = 0; i < _k->N; i++) B[i] = (double*) calloc (sizeof(double), _k->N);
 for (size_t i = 0; i < _k->N; i++) Btmp[i] = (double*) calloc (sizeof(double), _k->N);

 _initializedSample = (bool*) calloc (sizeof(bool), s_max);
 _fitnessVector = (double*) calloc (sizeof(double), s_max);

 // Init Generation
 terminate = false;
 _currentGeneration = 0;

 // Initializing Gaussian Generator
 auto jsGaussian = nlohmann::json();
 jsGaussian["Type"] = "Gaussian";
 jsGaussian["Mean"] = 0.0;
 jsGaussian["Sigma"] = 1.0;
 jsGaussian["Seed"] = _k->_seed++;
 _gaussianGenerator = new Variable();
 _gaussianGenerator->setDistribution(jsGaussian);

 _chiN = sqrt((double) _k->N) * (1. - 1./(4.*_k->N) + 1./(21.*_k->N*_k->N));

 // Initailizing Mu
 _muWeights = (double *) calloc (sizeof(double), mu_max);

 // Setting algorithm internal variables
 if (_name == "CCMA-ES") initInternals(_via_mu);
 else initInternals(_mu);

 // Setting eigensystem evaluation Frequency
 //if( _covarianceEigenEvalFreq < 0.0) _covarianceEigenEvalFreq = 1.0/_covarianceMatrixLearningRate/((double)_k->N)/10.0;

 flgEigensysIsUptodate = true;

 countevals      = 0;
 countinfeasible = 0;
 resampled       = 0;
 
 bestEver = -std::numeric_limits<double>::max();

 psL2 = 0.0;

 /* set rgxmean */
 for (size_t i = 0; i < _k->N; ++i)
 {
   if(_initialMeans[i] < _lowerBounds[i] || _initialMeans[i] > _upperBounds[i])
    fprintf(stderr,"[Korali] Warning: Initial Value (%.4f) for \'%s\' is out of bounds (%.4f-%.4f).\n",
            _initialMeans[i],
            _k->_variables[i]->_name.c_str(),
            _lowerBounds[i],
            _upperBounds[i]);
   rgxmean[i] = rgxold[i] = _initialMeans[i];
 }

 Z   = (double**) malloc (sizeof(double*) * s_max);
 BDZ = (double**) malloc (sizeof(double*) * s_max);
 for (size_t i = 0; i < s_max; i++) Z[i]   = (double*) calloc (sizeof(double), _k->N);
 for (size_t i = 0; i < s_max; i++) BDZ[i] = (double*) calloc (sizeof(double), _k->N);


 _numConstraints = _k->_fconstraints.size();
 if ( _name == "CCMA-ES" && _numConstraints < 1 ) { fprintf( stderr, "[Korali] Warning: No constraints provided, please use Solver Method 'CMA-ES' for optimized runtime.\n"); exit(-1); }
 if ( _name == "CMA-ES" && _numConstraints > 0 ) { fprintf( stderr, "[Korali] Error: Constraints provided, please use Solver Method 'CCMA-ES'.\n" ); exit(-1);}


 // CCMA-ES variables
 if (_name == "CCMA-ES")
 {
    bestValidIdx     = -1;
    countcevals      = 0;
    correctionsC     = 0;
    maxnumviolations = 0;
    numviolations    = (size_t*) calloc (sizeof(size_t), s_max);
    viabilityBounds  = (double*) calloc (sizeof(double), _numConstraints);

    sucRates = (double*) calloc (sizeof(double), _numConstraints);
    std::fill_n( sucRates, _numConstraints, 0.5);

    viabilityImprovement  = (bool*) calloc (sizeof(bool), s_max);
    viabilityIndicator    = (bool**) malloc (sizeof(bool*) * _numConstraints);
    constraintEvaluations = (double**) malloc (sizeof(double*) * _numConstraints);
    for (size_t c = 0; c < _numConstraints; ++c) viabilityIndicator[c]    = (bool*) calloc (sizeof(bool), s_max);
    for (size_t c = 0; c < _numConstraints; ++c) constraintEvaluations[c] = (double*) calloc (sizeof(double), s_max);

    v = (double**) malloc (sizeof(double*) * _numConstraints);
    for (size_t i = 0; i < _numConstraints; ++i) v[i] = (double*) calloc (sizeof(double), _k->N);

    besteverCeval = (double*) calloc (sizeof(double), _numConstraints);
 }

 // If state is defined:
 if (isDefined(js, {_name, "State"}))
 {
  setState(js);
  js[_name].erase("State");
 }

}

CMAES::~CMAES()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Korali::Solver::CMAES::getConfiguration(nlohmann::json& js)
{
 js["Solver"] = _name;

 js[_name]["Result Output Frequency"] = _resultOutputFrequency;
 
 js[_name]["Sample Count"]            = _s;
 js[_name]["Sigma Cumulation Factor"] = _sigmaCumulationFactor;
 js[_name]["Damp Factor"]             = _dampFactor;
 js[_name]["Objective"]               = _objective;
 js[_name]["Max Resamplings"]         = _maxResamplings;
 js[_name]["Sigma Bounded"]           = _isSigmaBounded;

 js[_name]["Mu"]["Value"]      = _mu;
 js[_name]["Mu"]["Type"]       = _muType;
 //js[_name]["Mu"]["Covariance"] = _muCovariance;

 js[_name]["Covariance Matrix"]["Cumulative Covariance"]           = _cumulativeCovariance;
 //js[_name]["Covariance Matrix"]["Learning Rate"]                   = _covarianceMatrixLearningRate;
 //js[_name]["Covariance Matrix"]["Eigenvalue Evaluation Frequency"] = _covarianceEigenEvalFreq;
 js[_name]["Covariance Matrix"]["Is Diagonal"]                     = _isdiag;

 // Variable information
 for (size_t i = 0; i < _k->N; i++)
 {
  js["Variables"][i][_name]["Lower Bound"]   = _lowerBounds[i];
  js["Variables"][i][_name]["Upper Bound"]   = _upperBounds[i];
  js["Variables"][i][_name]["Initial Mean"]  = _initialMeans[i];
  js["Variables"][i][_name]["Initial Standard Deviation"]         = _initialStdDevs[i];
  js["Variables"][i][_name]["Minimum Standard Deviation Changes"] = _minStdDevChanges[i];
  js["Variables"][i][_name]["Log Space"]     = _variableLogSpace[i];
 }

 js[_name]["Termination Criteria"]["Max Generations"]["Value"]           = _termCondMaxGenerations;
 js[_name]["Termination Criteria"]["Max Generations"]["Active"]          = _isTermCondMaxGenerations;
 js[_name]["Termination Criteria"]["Max Model Evaluations"]["Value"]     = _termCondMaxFitnessEvaluations;
 js[_name]["Termination Criteria"]["Max Model Evaluations"]["Active"]    = _isTermCondMaxFitnessEvaluations;
 js[_name]["Termination Criteria"]["Min Fitness"]["Value"]               = _termCondFitness;
 js[_name]["Termination Criteria"]["Min Fitness"]["Active"]              = _isTermCondFitness;
 js[_name]["Termination Criteria"]["Fitness Diff Threshold"]["Value"]    = _termCondFitnessDiffThreshold;
 js[_name]["Termination Criteria"]["Fitness Diff Threshold"]["Active"]   = _isTermCondFitnessDiffThreshold;
 js[_name]["Termination Criteria"]["Min Standard Deviation"]["Value"]    = _termCondMinDeltaX;
 js[_name]["Termination Criteria"]["Min Standard Deviation"]["Active"]   = _isTermCondMinDeltaX;
 js[_name]["Termination Criteria"]["Max Standard Deviation"]["Value"]    = _termCondTolUpXFactor;
 js[_name]["Termination Criteria"]["Max Standard Deviation"]["Active"]   = _isTermCondTolUpXFactor;
 js[_name]["Termination Criteria"]["Max Condition Covariance"]["Value"]  = _termCondCovCond;
 js[_name]["Termination Criteria"]["Max Condition Covariance"]["Active"] = _isTermCondCovCond;
 js[_name]["Termination Criteria"]["Min Step Size"]["Value"]             = _termCondMinStepFac;
 js[_name]["Termination Criteria"]["Min Step Size"]["Active"]            = _isTermCondMinStep;

// State Information
 js[_name]["State"]["Current Generation"]        = _currentGeneration;
 js[_name]["State"]["Sigma"]                     = sigma;
 js[_name]["State"]["BestEverFunctionValue"]     = bestEver;
 js[_name]["State"]["PreviousBestFunctionValue"] = prevBest;
 js[_name]["State"]["CurrentBestFunctionValue"]  = currentFunctionValue;
 js[_name]["State"]["PrevFunctionValue"]         = prevFunctionValue;
 js[_name]["State"]["MaxDiagonalCovariance"]     = maxdiagC;
 js[_name]["State"]["MinDiagonalCovariance"]     = mindiagC;
 js[_name]["State"]["MaxEigenvalue"]             = maxEW;
 js[_name]["State"]["MinEigenvalue"]             = minEW;
 js[_name]["State"]["EigenSystemUpToDate"]       = flgEigensysIsUptodate;
 js[_name]["State"]["EvaluationCount"]           = countevals;
 js[_name]["State"]["InfeasibleCount"]           = countinfeasible;
 js[_name]["State"]["ConjugateEvolutionPathL2"]  = psL2;

 for (size_t i = 0; i < _current_s; i++) js[_name]["State"]["Index"]          += index[i];
 for (size_t i = 0; i < _current_s; i++) js[_name]["State"]["FunctionValues"] += _fitnessVector[i];

 for (size_t i = 0; i < _k->N; i++) js[_name]["State"]["CurrentMeanVector"]      += rgxmean[i];
 for (size_t i = 0; i < _k->N; i++) js[_name]["State"]["PreviousMeanVector"]     += rgxold[i];
 for (size_t i = 0; i < _k->N; i++) js[_name]["State"]["BestEverVector"]         += rgxbestever[i];
 for (size_t i = 0; i < _k->N; i++) js[_name]["State"]["CurrentBestVector"]      += curBestVector[i];
 for (size_t i = 0; i < _k->N; i++) js[_name]["State"]["AxisLengths"]            += axisD[i];
 for (size_t i = 0; i < _k->N; i++) js[_name]["State"]["EvolutionPath"]          += rgpc[i];
 for (size_t i = 0; i < _k->N; i++) js[_name]["State"]["ConjugateEvolutionPath"] += rgps[i];

 for (size_t i = 0; i < _current_s; i++) for (size_t j = 0; j < _k->N; j++) js[_name]["State"]["Samples"][i][j] = _samplePopulation[i*_k->N + j];
 for (size_t i = 0; i < _k->N; i++) for (size_t j = 0; j <= i; j++) { js[_name]["State"]["CovarianceMatrix"][i][j] = C[i][j]; js[_name]["State"]["CovarianceMatrix"][j][i] = C[i][j]; }
 for (size_t i = 0; i < _k->N; i++) for (size_t j = 0; j <= i; j++) { js[_name]["State"]["EigenMatrix"][i][j] = B[i][j]; js[_name]["State"]["EigenMatrix"][j][i] = B[i][j]; }

 // CCMA-ES
 if (_name == "CCMA-ES")
 {
   js[_name]["Viability Regime"]             = _isViabilityRegime;
   js[_name]["Viability Sample Count"]       = _via_s;
   js[_name]["Mu"]["Viability"]              = _via_mu;
   js[_name]["Target Success Rate"]          = _targetSucRate;
   js[_name]["Adaption Size"]                = _adaptionSize;
   js[_name]["Max Corrections"]              = _maxCorrections;
   js[_name]["Normal Vector Learning Rate"]  = _cv;
   js[_name]["Global Success Learning Rate"] = _cp;

   // CCMA-ES States
   js[_name]["State"]["Global Success Rate"] = _globalSucRate;
   for (size_t c = 0; c < _numConstraints; ++c) js[_name]["State"]["Success Rates"][c] = sucRates[c];
   for (size_t c = 0; c < _numConstraints; ++c) js[_name]["State"]["Viability Boundaries"][c] = viabilityBounds[c];
   for (size_t i = 0; i < _current_s; ++i) js[_name]["State"]["Num Constraint Violations"][i] = numviolations[i];
   for (size_t c = 0; c < _numConstraints; ++c) for (size_t i = 0; i < _current_s; ++i) js[_name]["State"]["Constraint Evaluations"][c][i] = constraintEvaluations[c][i];
   for (size_t c = 0; c < _numConstraints; ++c) for (size_t d = 0; d < _k->N; ++d) js[_name]["State"]["Constraint Normal Approximation"][c][d] = v[c][d];
 }
}



void CMAES::setConfiguration(nlohmann::json& js)
{
 _resultOutputFrequency = consume(js, { _name, "Result Output Frequency" }, KORALI_NUMBER, std::to_string(1));
 
 _s                             = consume(js, { _name, "Sample Count" }, KORALI_NUMBER);
 _sigmaCumulationFactorIn       = consume(js, { _name, "Sigma Cumulation Factor" }, KORALI_NUMBER, std::to_string(-1));
 _dampFactorIn                  = consume(js, { _name, "Damp Factor" }, KORALI_NUMBER, std::to_string(-1));
 _objective                     = consume(js, { _name, "Objective" }, KORALI_STRING, "Maximize");
 _maxResamplings                = consume(js, { _name, "Max Resamplings" }, KORALI_NUMBER, std::to_string(1e6));
 _isSigmaBounded                = consume(js, { _name, "Sigma Bounded" }, KORALI_BOOLEAN, "false");


 _fitnessSign   = 0;
 if(_objective == "Maximize") _fitnessSign = 1;
 if(_objective == "Minimize") _fitnessSign = -1;
 if(_fitnessSign == 0)  { fprintf( stderr, "[Korali] %s Error: Invalid setting for Objective: %s\n", _name.c_str(), _objective.c_str()); exit(-1); }

 _mu                            = consume(js, { _name, "Mu", "Value" }, KORALI_NUMBER, std::to_string(ceil(_s / 2)));
 _muType                        = consume(js, { _name, "Mu", "Type" }, KORALI_STRING, "Logarithmic");
 _muCovarianceIn                = consume(js, { _name, "Mu", "Covariance" }, KORALI_NUMBER, std::to_string(-1));
 
 if( _mu < 1 || _mu > _s || ( ( _mu == _s )  && _muType.compare("Linear") ) )
   { fprintf( stderr, "[Korali] %s Error: Invalid setting of Mu (%lu) and/or Lambda (%lu)\n", _name.c_str(), _mu, _s); exit(-1); }

 // CCMA-ES (more below)
 if (_name == "CCMA-ES")
 {
    _via_s  = consume(js, { _name, "Viability Sample Count" }, KORALI_NUMBER);
    _via_mu = consume(js, { _name, "Mu", "Viability" }, KORALI_NUMBER, std::to_string(ceil(_via_s / 2)));
    if(_via_mu < 1 ||  _via_mu > _via_s || ( ( _via_mu == _via_s) && _muType.compare("Linear") ) )
        { fprintf( stderr, "[Korali] CCMA-ES Error: Invalid setting of Mu Viability (%lu) and/or Viability Sample Count (%lu)\n", _via_mu, _via_s); exit(-1); }
 }

 //_covarianceEigenEvalFreq      = consume(js, { _name, "Covariance Matrix", "Eigenvalue Evaluation Frequency" }, KORALI_NUMBER, std::to_string(0));
 _cumulativeCovarianceIn       = consume(js, { _name, "Covariance Matrix", "Cumulative Covariance" }, KORALI_NUMBER, std::to_string(-1));
 _covMatrixLearningRateIn      = consume(js, { _name, "Covariance Matrix", "Learning Rate" }, KORALI_NUMBER, std::to_string(-1));
 _isdiag                       = consume(js, { _name, "Covariance Matrix", "Is Diagonal" }, KORALI_BOOLEAN, "false");

 // Setting variable information
 _variableLogSpace = (bool*) calloc(sizeof(bool), _k->N);
 _initialMeans = (double*) calloc(sizeof(double), _k->N);
 _initialStdDevs = (double*) calloc(sizeof(double), _k->N);
 _minStdDevChanges = (double*) calloc(sizeof(double), _k->N);
 _lowerBounds = (double*) calloc(sizeof(double), _k->N);
 _upperBounds = (double*) calloc(sizeof(double), _k->N);

 for (size_t i = 0; i < _k->N; i++)
 {
  _lowerBounds[i] = -INFINITY;
  _upperBounds[i] = +INFINITY;

  _varNames.push_back(consume(js["Variables"][i], { "Name" }, KORALI_STRING, "X"+std::to_string(i)));

  bool lowerBoundDefined = isDefined(js["Variables"][i], { _name, "Lower Bound" });
  bool upperBoundDefined = isDefined(js["Variables"][i], { _name, "Upper Bound" });

  if (lowerBoundDefined) _lowerBounds[i] = consume(js["Variables"][i], { _name, "Lower Bound" }, KORALI_NUMBER);
  if (upperBoundDefined) _upperBounds[i] = consume(js["Variables"][i], { _name, "Upper Bound" }, KORALI_NUMBER);

  bool initialMeanDefined   = isDefined(js["Variables"][i], { _name, "Initial Mean" });
  bool initialStdDevDefined = isDefined(js["Variables"][i], { _name, "Initial Standard Deviation" });

  if (lowerBoundDefined && upperBoundDefined)
  {
   if (initialMeanDefined   == false) _initialMeans[i]   = (_upperBounds[i] + _lowerBounds[i]) * 0.5;
   if (initialStdDevDefined == false) _initialStdDevs[i] = (_upperBounds[i] - _lowerBounds[i]) * 0.30;
  }

  if (initialMeanDefined)   _initialMeans[i]   = consume(js["Variables"][i], { _name, "Initial Mean" }, KORALI_NUMBER);
  if (initialStdDevDefined) _initialStdDevs[i] = consume(js["Variables"][i], { _name, "Initial Standard Deviation" }, KORALI_NUMBER);

  _variableLogSpace[i] = consume(js["Variables"][i], { _name, "Log Space" }, KORALI_BOOLEAN, "false");
  _minStdDevChanges[i] = consume(js["Variables"][i], { _name, "Minimum Standard Deviation Changes" }, KORALI_NUMBER, std::to_string(0));
 }

 // Setting termination criteria
 _termCondMaxGenerations          = consume(js, { _name, "Termination Criteria", "Max Generations", "Value" }, KORALI_NUMBER, std::to_string(1000));
 _isTermCondMaxGenerations        = consume(js, { _name, "Termination Criteria", "Max Generations", "Active" }, KORALI_BOOLEAN, "true");
 _termCondMaxFitnessEvaluations   = consume(js, { _name, "Termination Criteria", "Max Model Evaluations", "Value" }, KORALI_NUMBER, std::to_string(std::numeric_limits<size_t>::max()));
 _isTermCondMaxFitnessEvaluations = consume(js, { _name, "Termination Criteria", "Max Model Evaluations", "Active" }, KORALI_BOOLEAN, "true");
 _termCondFitness                 = consume(js, { _name, "Termination Criteria", "Min Fitness", "Value" }, KORALI_NUMBER, std::to_string(std::numeric_limits<double>::max()));
 _isTermCondFitness               = consume(js, { _name, "Termination Criteria", "Min Fitness", "Active" }, KORALI_BOOLEAN, "false");
 _termCondFitnessDiffThreshold    = consume(js, { _name, "Termination Criteria", "Fitness Diff Threshold", "Value" }, KORALI_NUMBER, std::to_string(1e-9));
 _isTermCondFitnessDiffThreshold  = consume(js, { _name, "Termination Criteria", "Fitness Diff Threshold", "Active" }, KORALI_BOOLEAN, "true");
 _termCondMinDeltaX               = consume(js, { _name, "Termination Criteria", "Min Standard Deviation", "Value" }, KORALI_NUMBER, std::to_string(1e-12));
 _isTermCondMinDeltaX             = consume(js, { _name, "Termination Criteria", "Min Standard Deviation", "Active" }, KORALI_BOOLEAN, "false");
 _termCondTolUpXFactor            = consume(js, { _name, "Termination Criteria", "Max Standard Deviation", "Value" }, KORALI_NUMBER, std::to_string(1e18));
 _isTermCondTolUpXFactor          = consume(js, { _name, "Termination Criteria", "Max Standard Deviation", "Active" }, KORALI_BOOLEAN, "true");
 _termCondCovCond                 = consume(js, { _name, "Termination Criteria", "Max Condition Covariance", "Value" }, KORALI_NUMBER, std::to_string(1e18));
 _isTermCondCovCond               = consume(js, { _name, "Termination Criteria", "Max Condition Covariance", "Active" }, KORALI_BOOLEAN, "true");
 _termCondMinStepFac              = consume(js, { _name, "Termination Criteria", "Min Step Size", "Value" }, KORALI_NUMBER, std::to_string(0.2));
 _isTermCondMinStep               = consume(js, { _name, "Termination Criteria", "Min Step Size", "Active" }, KORALI_BOOLEAN, "true");

 if( (_name == "CCMA-ES") && _isViabilityRegime) {
     _current_s  = _via_s;
     _current_mu = _via_mu;
 } else {
     _current_s  = _s;
     _current_mu = _mu;
 }

 // CCMA-ES
 if (_name == "CCMA-ES")
 {
 _targetSucRate  = consume(js, { _name, "Target Success Rate" }, KORALI_NUMBER, std::to_string(2./11.));
 _adaptionSize   = consume(js, { _name, "Adaption Size" }, KORALI_NUMBER, std::to_string(0.1));
 _maxCorrections = consume(js, { _name, "Max Corrections" }, KORALI_NUMBER, std::to_string(1e6));
 _cv             = consume(js, { _name, "Normal Vector Learning Rate" }, KORALI_NUMBER, std::to_string(1.0/(2.0+_current_s)));
 _cp             = consume(js, { _name, "Global Success Learning Rate" }, KORALI_NUMBER, std::to_string(1.0/12.0));

 if( (_cv <= 0.0) || (_cv > 1.0) ) { fprintf( stderr, "[Korali] %s Error: Invalid Normal Vector Learning Rate (%f), must be greater 0.0 and less 1.0\n", _name.c_str(), _cv ); exit(-1); }
 if( (_cp <= 0.0) || (_cp > 1.0) ) { fprintf( stderr, "[Korali] %s Error: Invalid Global Success Learning Rate (%f), must be greater 0.0 and less 1.0\n", _name.c_str(), _cp ); exit(-1); }
 if( (_targetSucRate <= 0.0) || (_targetSucRate > 1.0) ) { fprintf( stderr, "[Korali] %s Error: Invalid Target Success Rate (%f), must be greater 0.0 and less 1.0\n", _name.c_str(), _targetSucRate ); exit(-1); }
 if(_adaptionSize <= 0.0) { fprintf( stderr, "[Korali] %s Error: Invalid Adaption Size (%f), must be greater 0.0\n", _name.c_str(), _adaptionSize ); exit(-1); }

 _isViabilityRegime = consume(js, { _name, "Viability Regime" }, KORALI_BOOLEAN, "true");
}

}


void CMAES::setState(nlohmann::json& js)
{
 _currentGeneration    = js[_name]["State"]["Current Generation"];
 sigma                 = js[_name]["State"]["Sigma"];
 bestEver              = js[_name]["State"]["BestEverFunctionValue"];
 currentFunctionValue  = js[_name]["State"]["CurrentBestFunctionValue"];
 prevBest              = js[_name]["State"]["PreviousBestFunctionValue"];
 prevFunctionValue     = js[_name]["State"]["PrevFunctionValue"];
 maxdiagC              = js[_name]["State"]["MaxDiagonalCovariance"];
 mindiagC              = js[_name]["State"]["MinDiagonalCovariance"];
 maxEW                 = js[_name]["State"]["MaxEigenvalue"];
 minEW                 = js[_name]["State"]["MinEigenvalue"] ;
 flgEigensysIsUptodate = js[_name]["State"]["EigenSystemUpToDate"];
 countevals            = js[_name]["State"]["EvaluationCount"];
 countinfeasible       = js[_name]["State"]["InfeasibleCount"];

 for (size_t i = 0; i < _k->N; i++) rgxmean[i]       = js[_name]["State"]["CurrentMeanVector"][i];
 for (size_t i = 0; i < _k->N; i++) rgxold[i]        = js[_name]["State"]["PreviousMeanVector"][i];
 for (size_t i = 0; i < _k->N; i++) rgxbestever[i]   = js[_name]["State"]["BestEverVector"][i];
 for (size_t i = 0; i < _k->N; i++) axisD[i]         = js[_name]["State"]["AxisLengths"][i];
 for (size_t i = 0; i < _k->N; i++) rgpc[i]          = js[_name]["State"]["EvolutionPath"][i];
 for (size_t i = 0; i < _k->N; i++) curBestVector[i] = js[_name]["State"]["CurrentBestVector"][i];
 for (size_t i = 0; i < _current_s; i++) index[i]              = js[_name]["State"]["Index"][i];
 for (size_t i = 0; i < _current_s; i++) _fitnessVector[i]     = js[_name]["State"]["FunctionValues"][i];
 for (size_t i = 0; i < _current_s; i++) for (size_t j = 0; j < _k->N; j++) _samplePopulation[i*_k->N + j] = js[_name]["State"]["Samples"][i][j];
 for (size_t i = 0; i < _k->N; i++) for (size_t j = 0; j < _k->N; j++) C[i][j] = js[_name]["State"]["CovarianceMatrix"][i][j];
 for (size_t i = 0; i < _k->N; i++) for (size_t j = 0; j < _k->N; j++) B[i][j] = js[_name]["State"]["EigenMatrix"][i][j];

 // CCMA-ES
 if ( _name == "CCMA-ES" )
 {
    for (size_t c = 0; c < _numConstraints; ++c) sucRates[c]        = js[_name]["State"]["Success Rates"][c];
    for (size_t c = 0; c < _numConstraints; ++c) viabilityBounds[c] = js[_name]["State"]["Viability Boundaries"][c];
    _globalSucRate = js[_name]["State"]["Global Success Rate"];

    for (size_t i = 0; i < _current_s; ++i) numviolations[i] = js[_name]["State"]["Num Constraint Violations"][i];

    for (size_t c = 0; c < _numConstraints; ++c) for (size_t i = 0; i < _current_s; ++i) constraintEvaluations[c][i] = js[_name]["State"]["Constraint Evaluations"][c][i];
    for (size_t c = 0; c < _numConstraints; ++c) for (size_t d = 0; d < _k->N; ++d) v[c][d] = js[_name]["State"]["Constraint Normal Approximation"][c][d];
 }
}


void CMAES::initInternals(size_t numsamplesmu)
{
 // Checking Variable values
 for (size_t i = 0; i < _k->N; i++)
 {
  if (_initialMeans[i] < _lowerBounds[i] || _initialMeans[i] > _upperBounds[i])
  { fprintf( stderr, "[Korali] %s Error: Initial Mean (%f) outside of Lower - Upper (%f - %f) bounds range defined for variable %d.\n", _name.c_str(), _initialMeans[i], _lowerBounds[i], _upperBounds[i], i); exit(-1); }

  if (_upperBounds[i] <= _lowerBounds[i])
  { fprintf( stderr, "[Korali] %s Error: Invalid Lower (%f) - Upper (%f) bounds range defined for variable %d.\n", _name.c_str(), _lowerBounds[i], _upperBounds[i], i); exit(-1); }

  if (_initialStdDevs[i] <= 0.0) { fprintf(stderr, "[Korali] %s Error: Initial StdDev for variable %d is less or equal 0.\n", _name.c_str(), i);  exit(-1); }
 }

 // Initializing Mu Weights
 if      (_muType == "Linear")      for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = numsamplesmu - i;
 else if (_muType == "Equal")       for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = 1;
 else if (_muType == "Logarithmic") for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = log(std::max( (double)numsamplesmu, 0.5*_current_s)+0.5)-log(i+1.);
 else  { fprintf( stderr, "[Korali] %s Error: Invalid setting of Mu Type (%s) (Linear, Equal or Logarithmic accepted).", _name.c_str(), _muType.c_str()); exit(-1); }

 // Normalize weights vector and set mueff
 double s1 = 0.0;
 double s2 = 0.0;

 for (size_t  i = 0; i < numsamplesmu; i++)
 {
  s1 += _muWeights[i];
  s2 += _muWeights[i]*_muWeights[i];
 }
 _muEffective = s1*s1/s2;

 for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] /= s1;

 // Setting Mu Covariance
 //if (_muCovarianceIn < 1) _muCovariance = _muEffective;
 //else                     _muCovariance = _muCovarianceIn;

 // Setting Cumulative Covariancea
 if( (_cumulativeCovarianceIn <= 0) || (_cumulativeCovarianceIn > 1) ) _cumulativeCovariance = (4.0 + _muEffective/(1.0*_k->N)) / (_k->N+4.0 + 2.0*_muEffective/(1.0*_k->N));
 else _cumulativeCovariance = _cumulativeCovarianceIn;

 /*
 // Setting Covariance Matrix Learning Rate
 double l1 = 2. / ((_k->N+1.4142)*(_k->N+1.4142));
 double l2 = (2.*_muEffective-1.) / ((_k->N+2.)*(_k->N+2.)+_muEffective);
 l2 = (l2 > 1) ? 1 : l2;
 l2 = (1./_muCovariance) * l1 + (1.-1./_muCovariance) * l2;

 _covarianceMatrixLearningRate = _covMatrixLearningRateIn;
 if (_covarianceMatrixLearningRate < 0 || _covarianceMatrixLearningRate > 1)  _covarianceMatrixLearningRate = l2;
 */

  // Setting Sigma Cumulation Factor
 _sigmaCumulationFactor = _sigmaCumulationFactorIn;
 if (_sigmaCumulationFactor <= 0 || _sigmaCumulationFactor >= 1) _sigmaCumulationFactor = (_muEffective + 2.0) / (_k->N + _muEffective + 3.0);

 // Setting Damping Factor
 _dampFactor = _dampFactorIn;
 if (_dampFactor <= 0.0)
     _dampFactor = (1.0 + 2*std::max(0.0, sqrt((_muEffective-1.0)/(_k->N+1.0)) - 1))  /* basic factor */
        // * std::max(0.3, 1. - (double)_k->N / (1e-6+std::min(_termCondMaxGenerations, _termCondMaxFitnessEvaluations/_via_s))) /* modification for short runs */
        + _sigmaCumulationFactor; /* minor increment */

 // Setting Sigma
 _trace = 0.0;
 for (size_t i = 0; i < _k->N; ++i) _trace += _initialStdDevs[i]*_initialStdDevs[i];
 sigma = sqrt(_trace/_k->N); /* _muEffective/(0.2*_muEffective+sqrt(_k->N)) * sqrt(_trace/_k->N); */

 // Setting B, C and axisD
 for (size_t i = 0; i < _k->N; ++i)
 {
  B[i][i] = 1.0;
  C[i][i] = axisD[i] = _initialStdDevs[i] * sqrt(_k->N / _trace);
  C[i][i] *= C[i][i];
 }

 minEW = doubleRangeMin(axisD, _k->N); minEW = minEW * minEW;
 maxEW = doubleRangeMax(axisD, _k->N); maxEW = maxEW * maxEW;

 maxdiagC=C[0][0]; for(size_t i=1;i<_k->N;++i) if(maxdiagC<C[i][i]) maxdiagC=C[i][i];
 mindiagC=C[0][0]; for(size_t i=1;i<_k->N;++i) if(mindiagC>C[i][i]) mindiagC=C[i][i];

 // CCMA-ES
 if ( _name == "CCMA-ES" )
 {
     // Setting beta
    _cv   = 1.0/(2.0+(double)_current_s);
    _beta = _adaptionSize/(_current_s+2.);
 }

}


/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/


void CMAES::run()
{
 if (_k->_verbosity >= KORALI_MINIMAL) {
   printf("[Korali] Starting %s (Objective: %s).\n", _name.c_str(), _objective.c_str());
   printf("--------------------------------------------------------------------\n");
 }

 startTime = std::chrono::system_clock::now();
 saveState();

 while(!checkTermination())
 {
   t0 = std::chrono::system_clock::now();
   if ( _name == "CCMA-ES" ) checkMeanAndSetRegime();
   prepareGeneration();
   evaluateSamples(); 
   if ( _name == "CCMA-ES" ){ updateConstraints(); handleConstraints(); }
   updateDistribution(_fitnessVector);
   _currentGeneration++;

   t1 = std::chrono::system_clock::now();

   printGeneration();
   saveState();
 }

 endTime = std::chrono::system_clock::now();

 printFinal();

}


void CMAES::evaluateSamples()
{
  double* transformedSamples = new double[ _current_s * _k->N ];
  
  for (size_t i = 0; i < _current_s; i++) for(size_t d = 0; d < _k->N; ++d)
    if(_variableLogSpace[d] == true) 
        transformedSamples[i*_k->N+d] = std::exp(_samplePopulation[i*_k->N+d]);
    else 
        transformedSamples[i*_k->N+d] = _samplePopulation[i*_k->N+d];


  while (_finishedSamples < _current_s)
  {
    for (size_t i = 0; i < _current_s; i++) if (_initializedSample[i] == false)
    {
      _initializedSample[i] = true; 
      _k->_conduit->evaluateSample(transformedSamples, i); countevals++;
    }
    _k->_conduit->checkProgress();
  }

  delete transformedSamples;
}


void CMAES::processSample(size_t sampleId, double fitness)
{
 double logPrior = _k->_problem->evaluateLogPrior(&_samplePopulation[sampleId*_k->N]);
 _fitnessVector[sampleId] = _fitnessSign * (logPrior+fitness);
 _finishedSamples++;
}


void CMAES::checkMeanAndSetRegime()
{
    if (_isViabilityRegime == false) return; /* mean already inside valid domain, no udpates */

    for (size_t c = 0; c < _numConstraints; ++c){
      countcevals++;
      std::vector<double> sample;
      for( size_t k=0; k<_k->N; k++) sample.push_back( rgxmean[k] );
      if ( _k->_fconstraints[c](sample) > 0.) return;
    } /* do nothing */

    /* mean inside domain, switch regime and update internal variables */
    _isViabilityRegime = false;

    for (size_t c = 0; c < _numConstraints; ++c) { viabilityBounds[c] = 0; } /* do nothing */
    _current_s  = _s;
    _current_mu = _mu;

    bestEver = -std::numeric_limits<double>::max();
    initInternals(_mu);

}


void CMAES::updateConstraints() //TODO: maybe parallelize constraint evaluations (DW)
{

 std::fill_n(numviolations, _current_s, 0);
 maxnumviolations = 0;

 for(size_t c = 0; c < _numConstraints; ++c)
 {
  double maxviolation = 0.0;
  for(size_t i = 0; i < _current_s; ++i)
  {
    countcevals++;
    std::vector<double> sample;
    for( size_t k=0; k<_k->N; k++) sample.push_back( _samplePopulation[i*_k->N+k] );

    constraintEvaluations[c][i] = _k->_fconstraints[c]( sample );

    if ( constraintEvaluations[c][i] > maxviolation ) maxviolation = constraintEvaluations[c][i];
    if ( _currentGeneration == 0 && _isViabilityRegime ) viabilityBounds[c] = maxviolation;

    if ( constraintEvaluations[c][i] > viabilityBounds[c] + 1e-12 ) numviolations[i]++;
    if ( numviolations[i] > maxnumviolations ) maxnumviolations = numviolations[i];

  }
 }

}


void CMAES::reEvaluateConstraints() //TODO: maybe we can parallelize constraint evaluations (DW)
{
  maxnumviolations = 0;
  for(size_t i = 0; i < _current_s; ++i) if(numviolations[i] > 0)
  {
    numviolations[i] = 0;
    for(size_t c = 0; c < _numConstraints; ++c)
    {
      countcevals++;
      std::vector<double> sample;
      for( size_t k=0; k<_k->N; k++) sample.push_back( _samplePopulation[i*_k->N+k] );

      constraintEvaluations[c][i] = _k->_fconstraints[c]( sample );

      if( constraintEvaluations[c][i] > viabilityBounds[c] + 1e-12 ) { viabilityIndicator[c][i] = true; numviolations[i]++; }
      else viabilityIndicator[c][i] = false;

    }
    if (numviolations[i] > maxnumviolations) maxnumviolations = numviolations[i];
  }
}


void CMAES::updateViabilityBoundaries()
{
 for(size_t c = 0; c < _numConstraints; ++c)
 {
  double maxviolation = 0.0;
  for(size_t i = 0; i < _current_mu /* _current_s alternative */ ; ++i) if (constraintEvaluations[c][index[i]] > maxviolation)
    maxviolation = constraintEvaluations[c][index[i]];

  viabilityBounds[c] = std::max( 0.0, std::min(viabilityBounds[c], 0.5*(maxviolation + viabilityBounds[c])) );
 }
}


bool CMAES::isFeasible(size_t sampleIdx) const
{
 for (size_t d = 0; d < _k->N; ++d)
  if (_samplePopulation[ sampleIdx*_k->N+d ] < _lowerBounds[d] || _samplePopulation[ sampleIdx*_k->N+d ] > _upperBounds[d]) return false;
 return true;
}


void CMAES::prepareGeneration()
{

 /* calculate eigensystem */
 updateEigensystem(C);

 for (size_t i = 0; i < _current_s; ++i)
 {
     size_t initial_infeasible = countinfeasible;
     sampleSingle(i);
     while( isFeasible( i ) == false )
     {
       countinfeasible++;
       sampleSingle(i);
       if ( (countinfeasible - initial_infeasible) > _maxResamplings )
       {
        if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: exiting resampling loop (param %zu) , max resamplings (%zu) reached.\n", i, _maxResamplings);
        exit(-1);
       }
     }
 }


 _finishedSamples = 0;
 for (size_t i = 0; i < _current_s; i++) _initializedSample[i] = false;
}


void CMAES::sampleSingle(size_t sampleIdx)
{

  /* generate scaled random vector (D * z) */
  for (size_t d = 0; d < _k->N; ++d)
  {
   Z[sampleIdx][d] = _gaussianGenerator->getRandomNumber();
   if (_isdiag) {
     BDZ[sampleIdx][d] = axisD[d] * Z[sampleIdx][d];
     _samplePopulation[sampleIdx * _k->N + d] = rgxmean[d] + sigma * BDZ[sampleIdx][d];
   }
   else rgdTmp[d] = axisD[d] * Z[sampleIdx][d];
  }

  if (!_isdiag)
   for (size_t d = 0; d < _k->N; ++d) {
    BDZ[sampleIdx][d] = 0.0;
    for (size_t e = 0; e < _k->N; ++e) BDZ[sampleIdx][d] += B[d][e] * rgdTmp[e];
    _samplePopulation[sampleIdx * _k->N + d] = rgxmean[d] + sigma * BDZ[sampleIdx][d];
  }
}


void CMAES::updateDistribution(const double *fitnessVector)
{

 /* Generate index */
 sort_index(fitnessVector, index, _current_s);

 if( (_name == "CMA-ES") ||  _isViabilityRegime )
  bestValidIdx = 0;
 else
 {
  bestValidIdx = -1;
  for (size_t i = 0; i < _current_s; i++) if(numviolations[index[i]] == 0) bestValidIdx = index[i];
  if ( bestValidIdx == -1 ) { printf("[Korali] Warning: all samples violate constraints, no updates taking place.\n"); return; }
 }

 /* update function value history */
 prevFunctionValue = currentFunctionValue;

 /* update current best */
 currentFunctionValue = fitnessVector[bestValidIdx];
 for (size_t d = 0; d < _k->N; ++d) curBestVector[d] = _samplePopulation[bestValidIdx*_k->N + d];

 /* update xbestever */
 //TODO: what if we minimize
 if ( currentFunctionValue > bestEver )
 {
  prevBest = bestEver;
  bestEver = currentFunctionValue;
  for (size_t d = 0; d < _k->N; ++d) rgxbestever[d]   = curBestVector[d];
  for (size_t c = 0; c < _numConstraints; ++c) besteverCeval[c] = constraintEvaluations[c][bestValidIdx];
 }

 histFuncValues[_currentGeneration] = bestEver;

 /* calculate rgxmean and rgBDz */
 for (size_t d = 0; d < _k->N; ++d) {
   rgxold[d] = rgxmean[d];
   rgxmean[d] = 0.;
   for (size_t i = 0; i < _current_mu; ++i)
     rgxmean[d] += _muWeights[i] * _samplePopulation[index[i]*_k->N + d];

   rgBDz[d] = (rgxmean[d] - rgxold[d])/sigma;
 }

 /* calculate z := D^(-1) * B^(T) * rgBDz into rgdTmp */
 for (size_t d = 0; d < _k->N; ++d) {
  double sum = 0.0;
  if (_isdiag) sum = rgBDz[d];
  else for (size_t e = 0; e < _k->N; ++e) sum += B[e][d] * rgBDz[e]; /* B^(T) * rgBDz ( iterating B[e][d] = B^(T) ) */

  rgdTmp[d] = sum / axisD[d]; /* D^(-1) * B^(T) * rgBDz */
 }

 psL2 = 0.0;

 /* cumulation for sigma (ps) using B*z */
 for (size_t d = 0; d < _k->N; ++d) {
    double sum = 0.0;
    if (_isdiag) sum = rgdTmp[d];
    else for (size_t e = 0; e < _k->N; ++e) sum += B[d][e] * rgdTmp[e];

    rgps[d] = (1. - _sigmaCumulationFactor) * rgps[d] + sqrt(_sigmaCumulationFactor * (2. - _sigmaCumulationFactor) * _muEffective) * sum;

    /* calculate norm(ps)^2 */
    psL2 += rgps[d] * rgps[d];
 }

 int hsig = sqrt(psL2) / sqrt(1. - pow(1.-_sigmaCumulationFactor, 2.0*(1.0+_currentGeneration))) / _chiN  < 1.4 + 2./(_k->N+1);

 /* cumulation for covariance matrix (pc) using B*D*z~_k->N(0,C) */
 for (size_t d = 0; d < _k->N; ++d)
   rgpc[d] = (1. - _cumulativeCovariance) * rgpc[d] + hsig * sqrt( _cumulativeCovariance * (2. - _cumulativeCovariance) * _muEffective ) * rgBDz[d];

 /* update of C  */
 adaptC(hsig);

 // update sigma & viability boundaries
 if( _name == "CCMA-ES" && (_isViabilityRegime == false) )
 {
   updateViabilityBoundaries();

   if ( prevBest == bestEver ) _globalSucRate = (1-_cp)*_globalSucRate;
   else _globalSucRate = (1-_cp)*_globalSucRate + _cp;
   //else for(size_t c = 0; c < _numConstraints; ++c) if( sucRates[c] < 0.5 ) { _globalSucRate = (1-_cp)*_globalSucRate; break; }
   sigma *= exp(1.0/_dampFactor*(_globalSucRate-(_targetSucRate/(1.0-_targetSucRate))*(1-_globalSucRate)));
   if(_k->_verbosity >= KORALI_DETAILED && sigma > 0.3) printf("[Korali] Warning: updateSigmaVie: sigma (%f) > 0.3\n", sigma);
   if(_k->_verbosity >= KORALI_DETAILED && sigma > 0.3) printf("[Korali] Warning: Sigma bounded \n");
   sigma = std::min( sigma, 0.3);
 }
 else
 {
   // sigma *= exp(((sqrt(psL2)/_chiN)-1.)*_sigmaCumulationFactor/_dampFactor) (orig, alternative)
   sigma *= exp(std::min(1.0, ((sqrt(psL2)/_chiN)-1.)*_sigmaCumulationFactor/_dampFactor));
 }

 /* upper bound for sigma */
 double sigma_upper = sqrt(_trace/_k->N);
 if( _isSigmaBounded && (sigma > sigma_upper) ) {
     fprintf(stderr, "[Korali] Warning: sigma bounded by %f, increase Initial Std of vairables.\n", sigma_upper);
     sigma = sigma_upper;
 }


 /* numerical error management */

 //treat maximal standard deviations
 //TODO

 //treat minimal standard deviations
 for (size_t d = 0; d < _k->N; ++d) if (sigma * sqrt(C[d][d]) < _minStdDevChanges[d])
 {
   sigma = (_minStdDevChanges[d])/sqrt(C[d][d]) * exp(0.05+_sigmaCumulationFactor/_dampFactor);
   if (_k->_verbosity >= KORALI_DETAILED) fprintf(stderr, "[Korali] Warning: sigma increased due to minimal standard deviation.\n");
 }

 //too low coordinate axis deviations
 //TODO

 //treat numerical precision provblems
 //TODO

 /* Test if function values are identical, escape flat fitness */
 if (currentFunctionValue == _fitnessVector[index[(int)_current_mu]]) {
   sigma *= exp(0.2+_sigmaCumulationFactor/_dampFactor);
   if (_k->_verbosity >= KORALI_DETAILED) {
     fprintf(stderr, "[Korali] Warning: sigma increased due to equal function values.\n");
   }
 }

 /*
 size_t horizon = 10 + ceil(3*10*_k->N/_s);
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

 //if (_covarianceMatrixLearningRate != 0.0)
 if (true)
 {
  /* definitions for speeding up inner-most loop */
  //double ccov1  = std::min(_covarianceMatrixLearningRate * (1./_muCovariance) * (_isdiag ? (_k->N+1.5) / 3. : 1.), 1.); (orig, alternative)
  //double ccovmu = std::min(_covarianceMatrixLearningRate * (1-1./_muCovariance) * (_isdiag ? (_k->N+1.5) / 3. : 1.), 1.-ccov1); (orig, alternative)
  double ccov1  = 2.0 / (std::pow(_k->N+1.3,2)+_muEffective);
  double ccovmu = std::min(1.0-ccov1,  2.0 * (_muEffective-2+1/_muEffective) / (std::pow(_k->N+2.0,2)+_muEffective));
  double sigmasquare = sigma * sigma;

  flgEigensysIsUptodate = false;

  /* update covariance matrix */
  for (size_t d = 0; d < _k->N; ++d)
   for (size_t e = _isdiag ? d : 0; e <= d; ++e) {
     C[d][e] = (1 - ccov1 - ccovmu) * C[d][e] + ccov1 * (rgpc[d] * rgpc[e] + (1-hsig)*ccov1*_cumulativeCovariance*(2.-_cumulativeCovariance) * C[d][e]);
     for (size_t k = 0; k < _current_mu; ++k) C[d][e] += ccovmu * _muWeights[k] * (_samplePopulation[index[k]*_k->N + d] - rgxold[d]) * (_samplePopulation[index[k]*_k->N + e] - rgxold[e]) / sigmasquare;
     if (e < d) C[e][d] = C[d][e];
   }

  /* update maximal and minimal diagonal value */
  maxdiagC = mindiagC = C[0][0];
  for (size_t d = 1; d < _k->N; ++d) {
  if (maxdiagC < C[d][d]) maxdiagC = C[d][d];
  else if (mindiagC > C[d][d])  mindiagC = C[d][d];
  }
 } /* if ccov... */
}


void CMAES::handleConstraints()
{
 size_t initial_resampled = resampled;
 size_t initial_corrections = correctionsC;

 while( maxnumviolations > 0 )
 {
  for (size_t i = 0; i < _k->N; i++) memcpy(Ctmp[i], C[i], sizeof(double) * _k->N);

  for(size_t i = 0; i < _current_s; ++i) if (numviolations[i] > 0)
  {
    //update v
    for( size_t c = 0; c < _numConstraints; ++c )
      if ( viabilityIndicator[c][i] == true )
      {
        correctionsC++;

        double v2 = 0;
        for( size_t d = 0; d < _k->N; ++d)
        {
            v[c][d] = (1.0-_cv)*v[c][d]+_cv*BDZ[i][d];
            v2 += v[c][d]*v[c][d];
        }
        for( size_t d = 0; d < _k->N; ++d)
          for( size_t e = 0; e < _k->N; ++e)
            Ctmp[d][e] = Ctmp[d][e] - ((_beta * _beta * v[c][d]*v[c][e])/(v2*numviolations[i]*numviolations[i]));

        flgEigensysIsUptodate = false;
        sucRates[c] = (1.0-_cp)*sucRates[c];
      }
      else
      {
        sucRates[c] = (1.0-_cp)*sucRates[c]+_cp/_current_s;
      }
   }

  updateEigensystem(Ctmp);

  /* in original some stopping criterion (TOLX) */
  // TODO

  //resample invalid points
  for(size_t i = 0; i < _current_s; ++i) if(numviolations[i] > 0)
  {
    do
    {
     resampled++;
     sampleSingle(i);
     if(resampled-initial_resampled > _maxResamplings)
     {
        if(_k->_verbosity >= KORALI_DETAILED) printf("[Korali] Warning: exiting resampling loop, max resamplings (%zu) reached.\n", _maxResamplings);
        reEvaluateConstraints();

        return;
     }

    }
    while( isFeasible(i) == false );
  }

  reEvaluateConstraints();

  if(correctionsC - initial_corrections > _maxCorrections)
  {
    if(_k->_verbosity >= KORALI_DETAILED) printf("[Korali] Warning: exiting adaption loop, max adaptions (%zu) reached.\n", _maxCorrections);
    return;
  }

 }//while maxnumviolations > 0

}


bool CMAES::checkTermination()
{

 if ( _isTermCondFitness && (_isViabilityRegime == false) && (_currentGeneration > 1) && (bestEver >= _termCondFitness) )
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
 size_t iTemp;
 for(iTemp=0; iTemp<_k->N; ++iTemp) {
  cTemp += (sigma * sqrt(C[iTemp][iTemp]) < _termCondMinDeltaX * _initialStdDevs[iTemp]) ? 1 : 0;
 }

 if ( _isTermCondMinDeltaX && (cTemp == _k->N) ) {
  terminate = true;
  sprintf(_terminationReason, "Object variable changes < %+6.3e", _termCondMinDeltaX * _initialStdDevs[iTemp]);
 }

 for(iTemp=0; iTemp<_k->N; ++iTemp)
  if ( _isTermCondTolUpXFactor && (sigma * sqrt(C[iTemp][iTemp]) > _termCondTolUpXFactor * _initialStdDevs[iTemp]) )
  {
    terminate = true;
    sprintf(_terminationReason, "Standard deviation increased by more than %7.2e, larger initial standard deviation recommended \n", _termCondTolUpXFactor * _initialStdDevs[iTemp]);
    break;
  }

 if ( _isTermCondCovCond && (maxEW >= minEW * _termCondCovCond) )
 {
   terminate = true;
   sprintf(_terminationReason, "Maximal condition number %7.2e reached. maxEW=%7.2e, minEig=%7.2e, maxdiagC=%7.2e, mindiagC=%7.2e\n",
                                _termCondCovCond, maxEW, minEW, maxdiagC, mindiagC);
 }

 double fac;
 size_t iAchse = 0;
 size_t iKoo = 0;
 if( _isTermCondMinStep )
 if (!_isdiag )
 {
    for (iAchse = 0; iAchse < _k->N; ++iAchse)
    {
    fac = _termCondMinStepFac * sigma * axisD[iAchse];
    for (iKoo = 0; iKoo < _k->N; ++iKoo){
      if (rgxmean[iKoo] != rgxmean[iKoo] + fac * B[iKoo][iAchse])
      break;
    }
    if (iKoo == _k->N)
    {
      terminate = true;
      sprintf(_terminationReason, "Standard deviation 0.1*%7.2e in principal axis %ld without effect.", fac/0.1, iAchse);
      break;
    } /* if (iKoo == _k->N) */
  } /* for iAchse    */
 } /* if _isdiag */

 /* Component of rgxmean is not changed anymore */
 if( _isTermCondMinStep )
 for (iKoo = 0; iKoo < _k->N; ++iKoo)
 {
  if (rgxmean[iKoo] == rgxmean[iKoo] + _termCondMinStepFac*sigma*sqrt(C[iKoo][iKoo]) ) //TODO: standard dev add to _isTermCond..
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


 eigen(_k->N, M, axisDtmp, Btmp);

 /* find largest and smallest eigenvalue, they are supposed to be sorted anyway */
 double minEWtmp = doubleRangeMin(axisDtmp, _k->N);
 double maxEWtmp = doubleRangeMax(axisDtmp, _k->N);

 if (minEWtmp <= 0.0)
  { printf("[Korali] Warning: Min Eigenvalue smaller or equal 0.0 (%+6.3e) after Eigen decomp (no update possible).\n", minEWtmp ); return; }

 /* write back */
 for (size_t d = 0; d < _k->N; ++d) axisD[d] = sqrt(axisDtmp[d]);
 for (size_t d = 0; d < _k->N; ++d) std::memcpy( B[d], Btmp[d], sizeof(double) * _k->N );
 for (size_t d = 0; d < _k->N; ++d) std::memcpy( C[d], M[d], sizeof(double) * _k->N );

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


void Korali::Solver::CMAES::saveState() const
{
 if (terminate || (_currentGeneration % _resultOutputFrequency) == 0) _k->saveState(_currentGeneration);
}


void CMAES::printGeneration() const
{
 if (_currentGeneration % _k->_outputFrequency != 0) return;

 if (_k->_verbosity >= KORALI_MINIMAL)
   printf("[Korali] Generation %ld - Duration: %fs (Total Elapsed Time: %fs)\n", _currentGeneration, std::chrono::duration<double>(t1-t0).count(), std::chrono::duration<double>(t1-startTime).count());

 if ( (_name == "CCMA-ES") && _isViabilityRegime && _k->_verbosity >= KORALI_NORMAL)
 {
   printf("\n[Korali] CCMA-ES searching start (MeanX violates constraints) .. \n");
   printf("[Korali] Viability Bounds:\n");
   for (size_t c = 0; c < _numConstraints; c++)  printf("         %s = (%+6.3e)\n", _k->_variables[c]->_name.c_str(), viabilityBounds[c]);
   printf("\n");
 }

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
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), rgxmean[d], rgxbestever[d]);

  if ( _name == "CCMA-ES" )
  if ( bestValidIdx >= 0 )
  {
    printf("[Korali] Constraint Evaluation at Current Function Value:\n");
      for (size_t c = 0; c < _numConstraints; ++c) printf("         ( %+6.3e )\n", constraintEvaluations[c][bestValidIdx]);
  }
  else
  {
    printf("[Korali] Constraint Evaluation at Current Best:\n");
    printf("[Korali] Warning: all samples violate constraints!\n");
      for (size_t c = 0; c < _numConstraints; ++c) printf("         ( %+6.3e )\n", constraintEvaluations[c][0]);
  }

  printf("[Korali] Covariance Matrix:\n");
  for (size_t d = 0; d < _k->N; d++)
  {
   for (size_t e = 0; e <= d; e++) printf("   %+6.3e  ",C[d][e]);
   printf("\n");
  }

  printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
  printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
  if ( _name == "CCMA-ES" ) { printf("[Korali] Number of Constraint Evaluations: %zu\n", countcevals); printf("[Korali] Number of Matrix Corrections: %zu\n", correctionsC ); }
 }

 if (_k->_verbosity >= KORALI_NORMAL)
   printf("--------------------------------------------------------------------\n");


}


void CMAES::printFinal() const
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("[Korali] %s Finished\n", _name.c_str());
    printf("[Korali] Optimum (%s) found: %e\n", _objective.c_str(), bestEver);
    printf("[Korali] Optimum (%s) found at:\n", _objective.c_str());
    for (size_t d = 0; d < _k->N; ++d) printf("         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), rgxbestever[d]);
    if ( _name == "CCMA-ES" ) { printf("[Korali] Constraint Evaluation at Optimum:\n"); for (size_t c = 0; c < _numConstraints; ++c) printf("         ( %+6.3e )\n", besteverCeval[c]); }
    printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
    printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
    printf("[Korali] Stopping Criterium: %s\n", _terminationReason);
    printf("[Korali] Total Elapsed Time: %fs\n", std::chrono::duration<double>(endTime-startTime).count());
    printf("--------------------------------------------------------------------\n");
 }
}
