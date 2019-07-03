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

CMAES::CMAES(nlohmann::json& js)
{
 
 setConfiguration(js);

 size_t s_max  = std::max(_s,  _via_s);
 size_t mu_max = std::max(_mu, _via_mu);

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

 _initializedSample = (bool*) calloc (sizeof(bool), s_max);
 _fitnessVector     = (double*) calloc (sizeof(double), s_max);

 // Init Generation
 _isFinished = false;
 _currentGeneration = 0;

 C    = (double**) calloc (sizeof(double*), _k->N);
 Ctmp = (double**) calloc (sizeof(double*), _k->N);
 B    = (double**) calloc (sizeof(double*), _k->N);
 Btmp = (double**) calloc (sizeof(double*), _k->N);
 for (size_t i = 0; i < _k->N; i++) C[i] = (double*) calloc (sizeof(double), _k->N);
 for (size_t i = 0; i < _k->N; i++) Ctmp[i] = (double*) calloc (sizeof(double), _k->N);
 for (size_t i = 0; i < _k->N; i++) B[i] = (double*) calloc (sizeof(double), _k->N);
 for (size_t i = 0; i < _k->N; i++) Btmp[i] = (double*) calloc (sizeof(double), _k->N);

 Z   = (double**) malloc (sizeof(double*) * s_max);
 BDZ = (double**) malloc (sizeof(double*) * s_max);
 for (size_t i = 0; i < s_max; i++) Z[i]   = (double*) calloc (sizeof(double), _k->N);
 for (size_t i = 0; i < s_max; i++) BDZ[i] = (double*) calloc (sizeof(double), _k->N);

 _transformedSamples = (double*) calloc (sizeof(double), s_max * _k->N);

 // Initailizing Mu
 _muWeights    = (double *) calloc (sizeof(double), mu_max);

 // Initializing Gaussian Generator
 auto jsGaussian = nlohmann::json();
 jsGaussian["Type"] = "Gaussian";
 jsGaussian["Mean"] = 0.0;
 jsGaussian["Sigma"] = 1.0;
 jsGaussian["Seed"] = _k->_seed++;
 _gaussianGenerator = new Variable();
 _gaussianGenerator->setDistribution(jsGaussian);

 _chiN = sqrt((double) _k->N) * (1. - 1./(4.*_k->N) + 1./(21.*_k->N*_k->N));
 
 // GSL Workspace
 gsl_eval  = gsl_vector_alloc(_k->N);
 gsl_evec  = gsl_matrix_alloc(_k->N, _k->N);
 gsl_work =  gsl_eigen_symmv_alloc(_k->N);

 _hasConstraints = (_k->_fconstraints.size() > 0);
 
 // CCMA-ES variables
 if (_hasConstraints)
 {
  if( (_cv <= 0.0) || (_cv > 1.0) ) { fprintf( stderr, "[Korali] CMA-ES Error: Invalid Normal Vector Learning Rate (%f), must be greater than 0.0 and less than 1.0\n", _cv ); exit(-1); }
  if( (_cp <= 0.0) || (_cp > 1.0) ) { fprintf( stderr, "[Korali] CMA-ES Error: Invalid Global Success Learning Rate (%f), must be greater than 0.0 and less than 1.0\n",  _cp ); exit(-1); }
  if( (_targetSucRate <= 0.0) || (_targetSucRate > 1.0) ) 
    { fprintf( stderr, "[Korali] CMA-ES Error: Invalid Target Success Rate (%f), must be greater than 0.0 and less than 1.0\n",  _targetSucRate ); exit(-1); }
  if(_adaptionSize <= 0.0) { fprintf( stderr, "[Korali] CMA-ES Error: Invalid Adaption Size (%f), must be greater than 0.0\n", _adaptionSize ); exit(-1); }

  bestValidIdx     = -1;
  countcevals      = 0;
  correctionsC     = 0;
  maxnumviolations = 0;
  numviolations    = (size_t*) calloc (sizeof(size_t), _current_s);
  viabilityBounds  = (double*) calloc (sizeof(double), _k->_fconstraints.size());

  sucRates = (double*) calloc (sizeof(double), _k->_fconstraints.size());
  std::fill_n( sucRates, _k->_fconstraints.size(), 0.5);

  viabilityImprovement  = (bool*) calloc (sizeof(bool), s_max);
  viabilityIndicator    = (bool**) calloc (sizeof(bool*), _k->_fconstraints.size());
  constraintEvaluations = (double**) calloc (sizeof(double*), _k->_fconstraints.size());

  for (size_t c = 0; c < _k->_fconstraints.size(); c++) viabilityIndicator[c]    = (bool*) calloc (sizeof(bool), s_max);
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) constraintEvaluations[c] = (double*) calloc (sizeof(double), s_max);

  v = (double**) calloc (sizeof(double*), _k->_fconstraints.size());
  for (size_t i = 0; i < _k->_fconstraints.size(); i++) v[i] = (double*) calloc (sizeof(double), _k->N);

  besteverCeval = (double*) calloc (sizeof(double), _k->_fconstraints.size());
 }

 // Setting algorithm internal variables
 if (_hasConstraints) { initInternals(_via_mu); initCovCorrectionParams(); }
 else initInternals(_mu);

 initCovariance();

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
   if(_initialMeans[i] < _lowerBounds[i] || _initialMeans[i] > _upperBounds[i]) if(_k->_verbosity >= KORALI_MINIMAL)
    fprintf(stderr,"[Korali] Warning: Initial Value (%.4f) of variable \'%s\' is out of bounds (%.4f-%.4f).\n",
            _initialMeans[i],
            _k->_variables[i]->_name.c_str(),
            _lowerBounds[i],
            _upperBounds[i]);
   rgxmean[i] = rgxold[i] = _initialMeans[i];
 }

 // If state is defined:
 if (isDefined(js, {"CMA-ES", "State"}))
 {
  setState(js);
  js["CMA-ES"].erase("State");
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
 js["Solver"] = "CMA-ES";

 js["CMA-ES"]["Result Output Frequency"] = _resultOutputFrequency;

 js["CMA-ES"]["Sample Count"]            = _s;
 js["CMA-ES"]["Sigma Cumulation Factor"] = _sigmaCumulationFactor;
 js["CMA-ES"]["Damp Factor"]             = _dampFactor;
 js["CMA-ES"]["Objective"]               = _objective;
 js["CMA-ES"]["Max Resamplings"]         = _maxResamplings;
 js["CMA-ES"]["Sigma Bounded"]           = _isSigmaBounded;

 js["CMA-ES"]["Mu"]["Value"]        = _mu;
 js["CMA-ES"]["Mu"]["Type"]         = _muType;
 //js["CMA-ES"]["Mu"]["Covariance"] = _muCovariance;

 js["CMA-ES"]["Covariance Matrix"]["Cumulative Covariance"]           = _cumulativeCovariance;
 js["CMA-ES"]["Covariance Matrix"]["Is Diagonal"]                     = _isdiag;
 //js["CMA-ES"]["Covariance Matrix"]["Eigenvalue Evaluation Frequency"] = _covarianceEigenEvalFreq;
 //js["CMA-ES"]["Covariance Matrix"]["Learning Rate"]                   = _covarianceMatrixLearningRate;

 // Variable information
 for (size_t i = 0; i < _k->N; i++)
 {
  js["Variables"][i]["CMA-ES"]["Lower Bound"]   = _lowerBounds[i];
  js["Variables"][i]["CMA-ES"]["Upper Bound"]   = _upperBounds[i];
  js["Variables"][i]["CMA-ES"]["Initial Mean"]  = _initialMeans[i];
  js["Variables"][i]["CMA-ES"]["Initial Standard Deviation"]         = _initialStdDevs[i];
  js["Variables"][i]["CMA-ES"]["Minimum Standard Deviation Changes"] = _minStdDevChanges[i];
  js["Variables"][i]["CMA-ES"]["Log Space"]     = _variableLogSpace[i];
 }

 js["CMA-ES"]["Termination Criteria"]["Max Generations"]["Value"]           = _termCondMaxGenerations;
 js["CMA-ES"]["Termination Criteria"]["Max Generations"]["Active"]          = _isTermCondMaxGenerations;
 js["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Value"]     = _termCondMaxFitnessEvaluations;
 js["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Active"]    = _isTermCondMaxFitnessEvaluations;
 js["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"]               = _termCondMinFitness;
 js["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Active"]              = _isTermCondMinFitness; 
 js["CMA-ES"]["Termination Criteria"]["Max Fitness"]["Value"]               = _termCondMaxFitness;
 js["CMA-ES"]["Termination Criteria"]["Max Fitness"]["Active"]              = _isTermCondMaxFitness;
 js["CMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"]["Value"]    = _termCondFitnessDiffThreshold;
 js["CMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"]["Active"]   = _isTermCondFitnessDiffThreshold;
 js["CMA-ES"]["Termination Criteria"]["Min Standard Deviation"]["Value"]    = _termCondMinDeltaX;
 js["CMA-ES"]["Termination Criteria"]["Min Standard Deviation"]["Active"]   = _isTermCondMinDeltaX;
 js["CMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Value"]    = _termCondTolUpXFactor;
 js["CMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Active"]   = _isTermCondTolUpXFactor;
 js["CMA-ES"]["Termination Criteria"]["Max Condition Covariance"]["Value"]  = _termCondCovCond;
 js["CMA-ES"]["Termination Criteria"]["Max Condition Covariance"]["Active"] = _isTermCondCovCond;

// State Information
 js["CMA-ES"]["State"]["Current Generation"]        = _currentGeneration;
 js["CMA-ES"]["State"]["Sigma"]                     = sigma;
 js["CMA-ES"]["State"]["BestEverFunctionValue"]     = bestEver;
 js["CMA-ES"]["State"]["PreviousBestFunctionValue"] = prevBest;
 js["CMA-ES"]["State"]["CurrentBestFunctionValue"]  = currentFunctionValue;
 js["CMA-ES"]["State"]["PrevFunctionValue"]         = prevFunctionValue;
 js["CMA-ES"]["State"]["MaxDiagonalCovariance"]     = maxdiagC;
 js["CMA-ES"]["State"]["MinDiagonalCovariance"]     = mindiagC;
 js["CMA-ES"]["State"]["MaxEigenvalue"]             = maxEW;
 js["CMA-ES"]["State"]["MinEigenvalue"]             = minEW;
 js["CMA-ES"]["State"]["EigenSystemUpToDate"]       = flgEigensysIsUptodate;
 js["CMA-ES"]["State"]["EvaluationCount"]           = countevals;
 js["CMA-ES"]["State"]["InfeasibleCount"]           = countinfeasible;
 js["CMA-ES"]["State"]["ConjugateEvolutionPathL2"]  = psL2;

 // CCMA-ES
 js["CMA-ES"]["Constraint"]["Viability"]["Regime"]          = _isViabilityRegime;
 js["CMA-ES"]["Constraint"]["Viability"]["Sample Count"]    = _via_s;
 js["CMA-ES"]["Constraint"]["Viability"]["Mu"]              = _via_mu;
 js["CMA-ES"]["Constraint"]["Target Success Rate"]          = _targetSucRate;
 js["CMA-ES"]["Constraint"]["Adaption Size"]                = _adaptionSize;
 js["CMA-ES"]["Constraint"]["Max Corrections"]              = _maxCorrections;
 js["CMA-ES"]["Constraint"]["Normal Vector Learning Rate"]  = _cv;
 js["CMA-ES"]["Constraint"]["Global Success Learning Rate"] = _cp;

 for (size_t i = 0; i < _current_s; i++) js["CMA-ES"]["State"]["Index"]          += index[i];
 for (size_t i = 0; i < _current_s; i++) js["CMA-ES"]["State"]["FunctionValues"] += _fitnessVector[i];

 for (size_t i = 0; i < _k->N; i++) js["CMA-ES"]["State"]["CurrentMeanVector"]      += rgxmean[i];
 for (size_t i = 0; i < _k->N; i++) js["CMA-ES"]["State"]["PreviousMeanVector"]     += rgxold[i];
 for (size_t i = 0; i < _k->N; i++) js["CMA-ES"]["State"]["BestEverVector"]         += rgxbestever[i];
 for (size_t i = 0; i < _k->N; i++) js["CMA-ES"]["State"]["CurrentBestVector"]      += curBestVector[i];
 for (size_t i = 0; i < _k->N; i++) js["CMA-ES"]["State"]["AxisLengths"]            += axisD[i];
 for (size_t i = 0; i < _k->N; i++) js["CMA-ES"]["State"]["EvolutionPath"]          += rgpc[i];
 for (size_t i = 0; i < _k->N; i++) js["CMA-ES"]["State"]["ConjugateEvolutionPath"] += rgps[i];

 for (size_t i = 0; i < _current_s; i++) for (size_t j = 0; j < _k->N; j++) js["CMA-ES"]["State"]["Samples"][i][j] = _samplePopulation[i*_k->N + j];
 for (size_t i = 0; i < _k->N; i++) for (size_t j = 0; j <= i; j++) { js["CMA-ES"]["State"]["CovarianceMatrix"][i][j] = C[i][j]; js["CMA-ES"]["State"]["CovarianceMatrix"][j][i] = C[i][j]; }
 for (size_t i = 0; i < _k->N; i++) for (size_t j = 0; j <= i; j++) { js["CMA-ES"]["State"]["EigenMatrix"][i][j] = B[i][j]; js["CMA-ES"]["State"]["EigenMatrix"][j][i] = B[i][j]; }

 // CCMA-ES States
 if (_hasConstraints)
 {
  js["CMA-ES"]["State"]["Constraint Global Success Rate"] = _globalSucRate;
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) js["CMA-ES"]["State"]["Constraint"]["Success Rates"][c] = sucRates[c];
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) js["CMA-ES"]["State"]["Constraint"]["Viability"]["Boundaries"][c] = viabilityBounds[c];
  for (size_t i = 0; i < _current_s; ++i) js["CMA-ES"]["State"]["Constraint"]["Violations"][i] = numviolations[i];
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) for (size_t i = 0; i < _current_s; ++i) js["CMA-ES"]["State"]["Constraint"]["Evaluations"][c][i] = constraintEvaluations[c][i];
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) for (size_t d = 0; d < _k->N; ++d) js["CMA-ES"]["State"]["Constraint"]["Normal Approximation"][c][d] = v[c][d];
 }
}


void CMAES::setConfiguration(nlohmann::json& js)
{
 _resultOutputFrequency = consume(js, { "CMA-ES", "Result Output Frequency" }, KORALI_NUMBER, std::to_string(1));
 
 _s                             = consume(js, { "CMA-ES", "Sample Count" }, KORALI_NUMBER);
 _sigmaCumulationFactorIn       = consume(js, { "CMA-ES", "Sigma Cumulation Factor" }, KORALI_NUMBER, std::to_string(-1));
 _dampFactorIn                  = consume(js, { "CMA-ES", "Damp Factor" }, KORALI_NUMBER, std::to_string(-1));
 _objective                     = consume(js, { "CMA-ES", "Objective" }, KORALI_STRING);
 _maxResamplings                = consume(js, { "CMA-ES", "Max Resamplings" }, KORALI_NUMBER, std::to_string(1e7));
 _isSigmaBounded                = consume(js, { "CMA-ES", "Sigma Bounded" }, KORALI_BOOLEAN, "false");

 _fitnessSign   = 0;
 if(_objective == "Maximize") _fitnessSign = 1;
 if(_objective == "Minimize") _fitnessSign = -1;
 if(_fitnessSign == 0)  { fprintf( stderr, "[Korali] CMA-ES Error: Invalid Objective: %s (choose Maximize or Minimize) \n", _objective.c_str()); exit(-1); }

 _mu                            = consume(js, { "CMA-ES", "Mu", "Value" }, KORALI_NUMBER, std::to_string(ceil(_s / 2)));
 _muType                        = consume(js, { "CMA-ES", "Mu", "Type" }, KORALI_STRING, "Logarithmic");
 _muCovarianceIn                = consume(js, { "CMA-ES", "Mu", "Covariance" }, KORALI_NUMBER, std::to_string(-1));
 
 if( _mu < 1 || _mu > _s || ( ( _mu == _s )  && _muType == "Linear") )
   { fprintf( stderr, "[Korali] CMA-ES Error: Invalid setting of Mu (%lu) and/or Lambda (%lu)\n",  _mu, _s); exit(-1); }

 // CCMA-ES (more below)
  _via_s  = consume(js, { "CMA-ES", "Constraint", "Viability", "Sample Count"}, KORALI_NUMBER, std::to_string(_s));
  _via_mu = consume(js, { "CMA-ES", "Constraint", "Viability", "Mu"}, KORALI_NUMBER, std::to_string(ceil(_via_s*0.5)));
  if(_via_mu < 1 ||  _via_mu > _via_s || ( ( _via_mu == _via_s) && _muType.compare("Linear") ) )
      { fprintf( stderr, "[Korali] CMA-ES Error: Invalid setting of Constraint Viability Mu (%lu) and/or Constraint Viability Sample Count (%lu)\n", _via_mu, _via_s); exit(-1); }

  _isViabilityRegime = consume(js, { "CMA-ES", "Constraint", "Viability", "Regime" }, KORALI_BOOLEAN, "false");

  if(_isViabilityRegime) {
      _current_s  = _via_s;
      _current_mu = _via_mu;
  } else {
      _current_s  = _s;
      _current_mu = _mu;
  }

 //_covarianceEigenEvalFreq      = consume(js, { "CMA-ES", "Covariance Matrix", "Eigenvalue Evaluation Frequency" }, KORALI_NUMBER, std::to_string(0));
 _cumulativeCovarianceIn       = consume(js, { "CMA-ES", "Covariance Matrix", "Cumulative Covariance" }, KORALI_NUMBER, std::to_string(-1));
 _covMatrixLearningRateIn      = consume(js, { "CMA-ES", "Covariance Matrix", "Learning Rate" }, KORALI_NUMBER, std::to_string(-1));
 _isdiag                       = consume(js, { "CMA-ES", "Covariance Matrix", "Is Diagonal" }, KORALI_BOOLEAN, "false");

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

  bool lowerBoundDefined = isDefined(js["Variables"][i], { "CMA-ES", "Lower Bound" });
  bool upperBoundDefined = isDefined(js["Variables"][i], { "CMA-ES", "Upper Bound" });

  if (lowerBoundDefined) _lowerBounds[i] = consume(js["Variables"][i], { "CMA-ES", "Lower Bound" }, KORALI_NUMBER);
  if (upperBoundDefined) _upperBounds[i] = consume(js["Variables"][i], { "CMA-ES", "Upper Bound" }, KORALI_NUMBER);

  bool initialMeanDefined   = isDefined(js["Variables"][i], { "CMA-ES", "Initial Mean" });
  bool initialStdDevDefined = isDefined(js["Variables"][i], { "CMA-ES", "Initial Standard Deviation" });

  if (lowerBoundDefined && upperBoundDefined)
  {
   if (initialMeanDefined   == false) _initialMeans[i]   = (_upperBounds[i] + _lowerBounds[i]) * 0.5;
   if (initialStdDevDefined == false) _initialStdDevs[i] = (_upperBounds[i] - _lowerBounds[i]) * 0.30;
  }

  if (initialMeanDefined)   _initialMeans[i]   = consume(js["Variables"][i], { "CMA-ES", "Initial Mean" }, KORALI_NUMBER);
  if (initialStdDevDefined) _initialStdDevs[i] = consume(js["Variables"][i], { "CMA-ES", "Initial Standard Deviation" }, KORALI_NUMBER);

  _variableLogSpace[i] = consume(js["Variables"][i], { "CMA-ES", "Log Space" }, KORALI_BOOLEAN, "false");
  _minStdDevChanges[i] = consume(js["Variables"][i], { "CMA-ES", "Minimum Standard Deviation Changes" }, KORALI_NUMBER, std::to_string(0));

  if ( (initialMeanDefined == false) && ((lowerBoundDefined == false) || (upperBoundDefined == false)) )
  { fprintf( stderr, "[Korali] CMA-ES Error: Initial Mean and Lower/Upper Bound not defined. Unable to init Initial Mean\n" ); exit(-1); }
  
  if ( (initialStdDevDefined == false) && ((lowerBoundDefined == false) || (upperBoundDefined == false)) )
  { fprintf( stderr, "[Korali] CMA-ES Error: Initial Standard Deviation and Lower/Upper Bound not defined. Unable to init Initial Standard Deviation\n" ); exit(-1); }
 }

 // Checking Variable values
 for (size_t i = 0; i < _k->N; i++)
 {
  if (_initialMeans[i] < _lowerBounds[i] || _initialMeans[i] > _upperBounds[i])
  { fprintf( stderr, "[Korali] CMA-ES Error: Initial Mean (%f) outside of Lower/Upper (%f - %f) Bound  defined for variable %d.\n",  _initialMeans[i], _lowerBounds[i], _upperBounds[i], i); exit(-1); }

  if (_upperBounds[i] <= _lowerBounds[i])
  { fprintf( stderr, "[Korali] CMA-ES Error: Invalid Lower/Upper (%f/%f) Bound defined for variable %d.\n",  _lowerBounds[i], _upperBounds[i], i); exit(-1); }

  if (_initialStdDevs[i] <= 0.0) { fprintf(stderr, "[Korali] CMA-ES Error: Initial Standard Deviation (%f) for variable %s is less or equal 0.\n", _initialStdDevs[i], _k->_variables[i]->_name.c_str()); exit(-1); }
 }

 // Setting termination criteria
 _termCondMaxGenerations          = consume(js, { "CMA-ES", "Termination Criteria", "Max Generations", "Value" }, KORALI_NUMBER, std::to_string(1000));
 _isTermCondMaxGenerations        = consume(js, { "CMA-ES", "Termination Criteria", "Max Generations", "Active" }, KORALI_BOOLEAN, "true");
 _termCondMaxFitnessEvaluations   = consume(js, { "CMA-ES", "Termination Criteria", "Max Model Evaluations", "Value" }, KORALI_NUMBER, std::to_string(std::numeric_limits<size_t>::max()));
 _isTermCondMaxFitnessEvaluations = consume(js, { "CMA-ES", "Termination Criteria", "Max Model Evaluations", "Active" }, KORALI_BOOLEAN, "true");
 _termCondMinFitness              = consume(js, { "CMA-ES", "Termination Criteria", "Min Fitness", "Value" }, KORALI_NUMBER, std::to_string(std::numeric_limits<double>::max()));
 _isTermCondMinFitness            = consume(js, { "CMA-ES", "Termination Criteria", "Min Fitness", "Active" }, KORALI_BOOLEAN, "false"); 
 _termCondMaxFitness              = consume(js, { "CMA-ES", "Termination Criteria", "Max Fitness", "Value" }, KORALI_NUMBER, std::to_string(-std::numeric_limits<double>::max()));
 _isTermCondMaxFitness            = consume(js, { "CMA-ES", "Termination Criteria", "Max Fitness", "Active" }, KORALI_BOOLEAN, "false");
 _termCondFitnessDiffThreshold    = consume(js, { "CMA-ES", "Termination Criteria", "Fitness Diff Threshold", "Value" }, KORALI_NUMBER, std::to_string(1e-9));
 _isTermCondFitnessDiffThreshold  = consume(js, { "CMA-ES", "Termination Criteria", "Fitness Diff Threshold", "Active" }, KORALI_BOOLEAN, "true");
 _termCondMinDeltaX               = consume(js, { "CMA-ES", "Termination Criteria", "Min Standard Deviation", "Value" }, KORALI_NUMBER, std::to_string(1e-12));
 _isTermCondMinDeltaX             = consume(js, { "CMA-ES", "Termination Criteria", "Min Standard Deviation", "Active" }, KORALI_BOOLEAN, "false");
 _termCondTolUpXFactor            = consume(js, { "CMA-ES", "Termination Criteria", "Max Standard Deviation", "Value" }, KORALI_NUMBER, std::to_string(1e18));
 _isTermCondTolUpXFactor          = consume(js, { "CMA-ES", "Termination Criteria", "Max Standard Deviation", "Active" }, KORALI_BOOLEAN, "true");
 _termCondCovCond                 = consume(js, { "CMA-ES", "Termination Criteria", "Max Condition Covariance", "Value" }, KORALI_NUMBER, std::to_string(1e18));
 _isTermCondCovCond               = consume(js, { "CMA-ES", "Termination Criteria", "Max Condition Covariance", "Active" }, KORALI_BOOLEAN, "true");
 _termCondMinStepFac              = consume(js, { "CMA-ES", "Termination Criteria", "Min Step Size Factor", "Value" }, KORALI_NUMBER, std::to_string(0.1));
 _isTermCondMinStepFac            = consume(js, { "CMA-ES", "Termination Criteria", "Min Step Size Factor", "Active" }, KORALI_BOOLEAN, "false");

 if( _isTermCondMinFitness && (_fitnessSign == 1) )
    { fprintf( stderr, "[Korali] CMA-ES Error: Invalid setting of Termination Criteria Min Fitness (objective is Maximize)\n"); exit(-1); }

 if( _isTermCondMaxFitness && (_fitnessSign == -1) )
    { fprintf( stderr, "[Korali] CMA-ES Error: Invalid setting of Termination Criteria Max Fitness (objective is Minimize)\n"); exit(-1); }

 // CCMA-ES
 _targetSucRate  = consume(js, { "CMA-ES", "Constraint", "Target Success Rate" }, KORALI_NUMBER, std::to_string(2./11.));
 _adaptionSize   = consume(js, { "CMA-ES", "Constraint", "Adaption Size" }, KORALI_NUMBER, std::to_string(0.1));
 _maxCorrections = consume(js, { "CMA-ES", "Constraint", "Max Corrections" }, KORALI_NUMBER, std::to_string(1e6));
 _cv             = consume(js, { "CMA-ES", "Constraint", "Normal Vector Learning Rate" }, KORALI_NUMBER, std::to_string(1.0/(2.0+_current_s)));
 _cp             = consume(js, { "CMA-ES", "Constraint", "Global Success Learning Rate" }, KORALI_NUMBER, std::to_string(1.0/12.0));
}

void CMAES::setState(nlohmann::json& js)
{
 _currentGeneration    = js["CMA-ES"]["State"]["Current Generation"];
 sigma                 = js["CMA-ES"]["State"]["Sigma"];
 bestEver              = js["CMA-ES"]["State"]["BestEverFunctionValue"];
 currentFunctionValue  = js["CMA-ES"]["State"]["CurrentBestFunctionValue"];
 prevBest              = js["CMA-ES"]["State"]["PreviousBestFunctionValue"];
 prevFunctionValue     = js["CMA-ES"]["State"]["PrevFunctionValue"];
 maxdiagC              = js["CMA-ES"]["State"]["MaxDiagonalCovariance"];
 mindiagC              = js["CMA-ES"]["State"]["MinDiagonalCovariance"];
 maxEW                 = js["CMA-ES"]["State"]["MaxEigenvalue"];
 minEW                 = js["CMA-ES"]["State"]["MinEigenvalue"] ;
 flgEigensysIsUptodate = js["CMA-ES"]["State"]["EigenSystemUpToDate"];
 countevals            = js["CMA-ES"]["State"]["EvaluationCount"];
 countinfeasible       = js["CMA-ES"]["State"]["InfeasibleCount"];

 for (size_t i = 0; i < _k->N; i++) rgxmean[i]       = js["CMA-ES"]["State"]["CurrentMeanVector"][i];
 for (size_t i = 0; i < _k->N; i++) rgxold[i]        = js["CMA-ES"]["State"]["PreviousMeanVector"][i];
 for (size_t i = 0; i < _k->N; i++) rgxbestever[i]   = js["CMA-ES"]["State"]["BestEverVector"][i];
 for (size_t i = 0; i < _k->N; i++) axisD[i]         = js["CMA-ES"]["State"]["AxisLengths"][i];
 for (size_t i = 0; i < _k->N; i++) rgpc[i]          = js["CMA-ES"]["State"]["EvolutionPath"][i];
 for (size_t i = 0; i < _k->N; i++) curBestVector[i] = js["CMA-ES"]["State"]["CurrentBestVector"][i];
 
 for (size_t i = 0; i < _current_s; i++) index[i]          = js["CMA-ES"]["State"]["Index"][i];
 for (size_t i = 0; i < _current_s; i++) _fitnessVector[i] = js["CMA-ES"]["State"]["FunctionValues"][i];
 
 for (size_t i = 0; i < _k->N; i++) for (size_t j = 0; j < _k->N; j++) C[i][j] = js["CMA-ES"]["State"]["CovarianceMatrix"][i][j];
 for (size_t i = 0; i < _k->N; i++) for (size_t j = 0; j < _k->N; j++) B[i][j] = js["CMA-ES"]["State"]["EigenMatrix"][i][j];

 for (size_t i = 0; i < _current_s; i++) for (size_t j = 0; j < _k->N; j++) _samplePopulation[i*_k->N + j] = js["CMA-ES"]["State"]["Samples"][i][j];
 
 // CCMA-ES
 if (_hasConstraints)
 {
  _globalSucRate = js["CMA-ES"]["State"]["Constraint Global Success Rate"];
  
  for (size_t i = 0; i < _current_s; ++i) numviolations[i] = js["CMA-ES"]["State"]["Constraint"]["Violations"][i];
  
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) sucRates[c]        = js["CMA-ES"]["State"]["Constraint"]["Success Rates"][c];
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) viabilityBounds[c] = js["CMA-ES"]["State"]["Constraint"]["Viability"]["Boundaries"][c];
  
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) for (size_t d = 0; d < _k->N; ++d) v[c][d] = js["CMA-ES"]["State"]["Constraint"]["Normal Approximation"][c][d];
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) for (size_t i = 0; i < _current_s; ++i) constraintEvaluations[c][i] = js["CMA-ES"]["State"]["Constraint"]["Evaluations"][c][i];
 }
}


void CMAES::initInternals(size_t numsamplesmu)
{

 // Initializing Mu Weights
 if      (_muType == "Linear")       for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = numsamplesmu - i;
 else if (_muType == "Equal")        for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = 1.;
 else if (_muType == "Logarithmic")  for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = log(std::max( (double)numsamplesmu, 0.5*_current_s)+0.5)-log(i+1.);
 else  { fprintf( stderr, "[Korali] CMA-ES Error: Invalid setting of Mu Type (%s) (Linear, Equal, or Logarithmic accepted).",  _muType.c_str()); exit(-1); }

 // Normalize weights vector and set mueff
 double s1 = 0.0;
 double s2 = 0.0;

 for (size_t i = 0; i < numsamplesmu; i++)
 {
  s1 += _muWeights[i];
  s2 += _muWeights[i]*_muWeights[i];
 }
 _muEffective = s1*s1/s2;

 for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] /= s1;

 // Setting Mu Covariance
 //if (_muCovarianceIn < 1) _muCovariance = _muEffective;
 //else                     _muCovariance = _muCovarianceIn;
 
 /*
 // Setting Covariance Matrix Learning Rate
 double l1 = 2. / ((_k->N+1.4142)*(_k->N+1.4142));
 double l2 = (2.*_muEffective-1.) / ((_k->N+2.)*(_k->N+2.)+_muEffective);
 l2 = (l2 > 1) ? 1 : l2;
 l2 = (1./_muCovariance) * l1 + (1.-1./_muCovariance) * l2;

 _covarianceMatrixLearningRate = _covMatrixLearningRateIn;
 if (_covarianceMatrixLearningRate < 0 || _covarianceMatrixLearningRate > 1)  _covarianceMatrixLearningRate = l2;
 */

 // Setting Cumulative Covariancea
 if( (_cumulativeCovarianceIn <= 0) || (_cumulativeCovarianceIn > 1) ) _cumulativeCovariance = (4.0 + _muEffective/(1.0*_k->N)) / (_k->N+4.0 + 2.0*_muEffective/(1.0*_k->N));
 else _cumulativeCovariance = _cumulativeCovarianceIn;

 // Setting Sigma Cumulation Factor
 _sigmaCumulationFactor = _sigmaCumulationFactorIn;
 if (_sigmaCumulationFactor <= 0 || _sigmaCumulationFactor >= 1) _sigmaCumulationFactor = (_muEffective + 2.0) / (_k->N + _muEffective + 3.0);

 // Setting Damping Factor
 _dampFactor = _dampFactorIn;
 if (_dampFactor <= 0.0)
     _dampFactor = (1.0 + 2*std::max(0.0, sqrt((_muEffective-1.0)/(_k->N+1.0)) - 1))  /* basic factor */
        // * std::max(0.3, 1. - (double)_k->N / (1e-6+std::min(_termCondMaxGenerations, _termCondMaxFitnessEvaluations/_via_s))) /* modification for short runs */
        + _sigmaCumulationFactor; /* minor increment */

}

void CMAES::initCovCorrectionParams()
{
  // Setting beta
  _cv   = 1.0/(2.0+(double)_current_s);
  _beta = _adaptionSize/(_current_s+2.);
}

void CMAES::initCovariance()
{
 
 // Setting Sigma
 _trace = 0.0;
 for (size_t i = 0; i < _k->N; ++i) _trace += _initialStdDevs[i]*_initialStdDevs[i];
 sigma = sqrt(_trace/_k->N);

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

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/


void CMAES::run()
{
 if (_k->_verbosity >= KORALI_MINIMAL) {
   printf("[Korali] Starting CMA-ES (Objective: %s).\n",  _objective.c_str());
   printf("--------------------------------------------------------------------\n");
 }

 startTime = std::chrono::system_clock::now();
 if ( _hasConstraints ) { checkMeanAndSetRegime(); updateConstraints(); }
 printGeneration();
 saveState();
 
 while(!checkTermination())
 {
   t0 = std::chrono::system_clock::now();
   
   prepareGeneration();
   evaluateSamples(); 
   
   if ( _hasConstraints ){ updateConstraints(); handleConstraints(); }
   updateDistribution(_fitnessVector);
   if ( _hasConstraints ) checkMeanAndSetRegime();
   
   _currentGeneration++;

   t1 = std::chrono::system_clock::now();

   printGeneration();
   saveState();
 }

 endTime = std::chrono::system_clock::now();

 saveState();
 printFinal();

}


void CMAES::evaluateSamples()
{
  for (size_t i = 0; i < _current_s; i++) for(size_t d = 0; d < _k->N; ++d)
    if(_variableLogSpace[d] == true) 
        _transformedSamples[i*_k->N+d] = std::exp(_samplePopulation[i*_k->N+d]);
    else 
        _transformedSamples[i*_k->N+d] = _samplePopulation[i*_k->N+d];

  while (_finishedSamples < _current_s)
  {
    for (size_t i = 0; i < _current_s; i++) if (_initializedSample[i] == false)
    {
      _initializedSample[i] = true; 
      _k->_conduit->evaluateSample(_transformedSamples, i); countevals++;
    }
    _k->_conduit->checkProgress();
  }
}


void CMAES::processSample(size_t sampleId, double fitness)
{
 double logPrior = _k->_problem->evaluateLogPrior(&_samplePopulation[sampleId*_k->N]);
 fitness = _fitnessSign * (logPrior+fitness);
 if(std::isfinite(fitness) == false)
 {
   fitness = _fitnessSign * std::numeric_limits<double>::max();
   printf("[Korali] Warning: sample %zu returned non finite fitness (set to %e)!\n", sampleId, fitness);
 }
 
 _fitnessVector[sampleId] = fitness;
 _finishedSamples++;
}


void CMAES::checkMeanAndSetRegime()
{
    if (_isViabilityRegime == false) return; /* mean already inside valid domain, no udpates */

    for (size_t c = 0; c < _k->_fconstraints.size(); c++){
      countcevals++;
      std::vector<double> sample(rgxmean, rgxmean+_k->N);
      if ( _k->_fconstraints[c](sample) > 0.) return; /* mean violates constraint, do nothing */
    }

    /* mean inside domain, switch regime and update internal variables */
    _isViabilityRegime = false;

    for (size_t c = 0; c < _k->_fconstraints.size(); c++) { viabilityBounds[c] = 0; }
    _current_s  = _s;
    _current_mu = _mu;

    bestEver = -std::numeric_limits<double>::max();
    initInternals(_current_mu);
    initCovCorrectionParams();
    initCovariance();
}


void CMAES::updateConstraints() //TODO: maybe parallelize constraint evaluations (DW)
{

 std::fill_n(numviolations, _current_s, 0);
 maxnumviolations = 0;

 for(size_t c = 0; c < _k->_fconstraints.size(); c++)
 {
  double maxviolation = 0.0;
  for(size_t i = 0; i < _current_s; ++i)
  {
    countcevals++;
    std::vector<double> sample(&_samplePopulation[i*_k->N], &_samplePopulation[(i+1)*_k->N]);

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
    for(size_t c = 0; c < _k->_fconstraints.size(); c++)
    {
      countcevals++;
      std::vector<double> sample(&_samplePopulation[i*_k->N], &_samplePopulation[(i+1)*_k->N]);

      constraintEvaluations[c][i] = _k->_fconstraints[c]( sample );

      if( constraintEvaluations[c][i] > viabilityBounds[c] + 1e-12 ) { viabilityIndicator[c][i] = true; numviolations[i]++; }
      else viabilityIndicator[c][i] = false;

    }
    if (numviolations[i] > maxnumviolations) maxnumviolations = numviolations[i];
  }
}


void CMAES::updateViabilityBoundaries()
{
 for(size_t c = 0; c < _k->_fconstraints.size(); c++)
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
 for (size_t d = 0; d < _k->N; ++d) memcpy(Ctmp[d], C[d], sizeof(double) * _k->N );
 updateEigensystem(Ctmp);

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
        if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: Exiting resampling loop (sample %zu), max resamplings (%zu) reached.\n", i, _maxResamplings);
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

 if( (_hasConstraints == false) ||  _isViabilityRegime )
  bestValidIdx = 0;
 else
 {
  bestValidIdx = -1;
  for (size_t i = 0; i < _current_s; i++) if(numviolations[index[i]] == 0) bestValidIdx = index[i];
  if(_k->_verbosity >= KORALI_DETAILED && bestValidIdx == -1) 
    { printf("[Korali] Warning: all samples violate constraints, no updates taking place.\n"); return; }
 }

 /* update function value history */
 prevFunctionValue = currentFunctionValue;

 /* update current best */
 currentFunctionValue = fitnessVector[bestValidIdx];
 for (size_t d = 0; d < _k->N; ++d) curBestVector[d] = _samplePopulation[bestValidIdx*_k->N + d];

 /* update xbestever */
 if ( currentFunctionValue > bestEver )
 {
  prevBest = bestEver;
  bestEver = currentFunctionValue;
  for (size_t d = 0; d < _k->N; ++d) rgxbestever[d]   = curBestVector[d];
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) besteverCeval[c] = constraintEvaluations[c][bestValidIdx];
 }

 histFuncValues[_currentGeneration] = bestEver;
 
 /* set weights */
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

 double sigma_upper = sqrt(_trace/_k->N);
 // update sigma & viability boundaries
 if( _hasConstraints && (_isViabilityRegime == false) )
 {
   updateViabilityBoundaries();

   if ( prevBest == bestEver ) _globalSucRate = (1-_cp)*_globalSucRate;
   else _globalSucRate = (1-_cp)*_globalSucRate + _cp;
   //else for(size_t c = 0; c < _k->_fconstraints.size(); c++) if( sucRates[c] < 0.5 ) { _globalSucRate = (1-_cp)*_globalSucRate; break; }
   sigma *= exp(1.0/_dampFactor*(_globalSucRate-(_targetSucRate/(1.0-_targetSucRate))*(1-_globalSucRate)));
 }
 else
 {
   // sigma *= exp(((sqrt(psL2)/_chiN)-1.)*_sigmaCumulationFactor/_dampFactor) (orig, alternative)
   sigma *= exp(std::min(1.0, ((sqrt(psL2)/_chiN)-1.)*_sigmaCumulationFactor/_dampFactor));
 }

 if(_k->_verbosity >= KORALI_DETAILED && sigma > sigma_upper) 
     printf("[Korali] Warning: Sigma exceeding inital value of sigma (%f > %f), increase Initial Standard Deviation of variables.\n", sigma, sigma_upper);
 
 /* upper bound for sigma */
 if( _isSigmaBounded && (sigma > sigma_upper) ) sigma = sigma_upper;

 /* numerical error management */

 //treat maximal standard deviations
 //TODO

 //treat minimal standard deviations
 for (size_t d = 0; d < _k->N; ++d) if (sigma * sqrt(C[d][d]) < _minStdDevChanges[d])
 {
   sigma = (_minStdDevChanges[d])/sqrt(C[d][d]) * exp(0.05+_sigmaCumulationFactor/_dampFactor);
   if (_k->_verbosity >= KORALI_DETAILED) fprintf(stderr, "[Korali] Warning: Sigma increased due to minimal standard deviation.\n");
 }

 //too low coordinate axis deviations
 //TODO

 //treat numerical precision provblems
 //TODO

 /* Test if function values are identical, escape flat fitness */
 if (currentFunctionValue == _fitnessVector[index[(int)_current_mu]]) {
   sigma *= exp(0.2+_sigmaCumulationFactor/_dampFactor);
   if (_k->_verbosity >= KORALI_DETAILED) {
     fprintf(stderr, "[Korali] Warning: Sigma increased due to equal function values.\n");
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
     for (size_t k = 0; k < _current_mu; ++k) 
         C[d][e] += ccovmu * _muWeights[k] * (_samplePopulation[index[k]*_k->N + d] - rgxold[d]) * (_samplePopulation[index[k]*_k->N + e] - rgxold[e]) / sigmasquare;
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
    for( size_t c = 0; c < _k->_fconstraints.size(); c++ )
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
        if(_k->_verbosity >= KORALI_DETAILED) printf("[Korali] Warning: Exiting resampling loop, max resamplings (%zu) reached.\n", _maxResamplings);
        reEvaluateConstraints();

        return;
     }

    }
    while( isFeasible(i) == false );
  }

  reEvaluateConstraints();

  if(correctionsC - initial_corrections > _maxCorrections)
  {
    if(_k->_verbosity >= KORALI_DETAILED) printf("[Korali] Warning: Exiting adaption loop, max adaptions (%zu) reached.\n", _maxCorrections);
    return;
  }

 }//while maxnumviolations > 0

}

bool CMAES::checkTermination()
{

 if ( _isTermCondMinFitness && (_isViabilityRegime == false) && (_currentGeneration > 1) && (bestEver >= _termCondMinFitness) )
 {
  _isFinished = true;
  sprintf(_terminationReason, "Min fitness value (%+6.3e) > (%+6.3e)",  bestEver, _termCondMinFitness);
 }
 
 if ( _isTermCondMaxFitness && (_isViabilityRegime == false) && (_currentGeneration > 1) && (bestEver >= _termCondMaxFitness) )
 {
  _isFinished = true;
  sprintf(_terminationReason, "Max fitness value (%+6.3e) > (%+6.3e)",  bestEver, _termCondMaxFitness);
 }

 double range = fabs(currentFunctionValue - prevFunctionValue);
 if ( _isTermCondFitnessDiffThreshold && (_currentGeneration > 1) && (range <= _termCondFitnessDiffThreshold) )
 {
  _isFinished = true;
  sprintf(_terminationReason, "Function value differences (%+6.3e) < (%+6.3e)",  range, _termCondFitnessDiffThreshold);
 }

 size_t cTemp = 0;
 size_t iTemp;
 for(iTemp=0; iTemp<_k->N; ++iTemp) {
  cTemp += (sigma * sqrt(C[iTemp][iTemp]) < _termCondMinDeltaX * _initialStdDevs[iTemp]) ? 1 : 0;
 }

 if ( _isTermCondMinDeltaX && (cTemp == _k->N) ) {
  _isFinished = true;
  sprintf(_terminationReason, "Object variable changes < %+6.3e", _termCondMinDeltaX * _initialStdDevs[iTemp]);
 }

 for(iTemp=0; iTemp<_k->N; ++iTemp)
  if ( _isTermCondTolUpXFactor && (sigma * sqrt(C[iTemp][iTemp]) > _termCondTolUpXFactor * _initialStdDevs[iTemp]) )
  {
    _isFinished = true;
    sprintf(_terminationReason, "Standard deviation increased by more than %7.2e, larger initial standard deviation recommended \n", _termCondTolUpXFactor * _initialStdDevs[iTemp]);
    break;
  }

 if ( _isTermCondCovCond && (maxEW >= minEW * _termCondCovCond) )
 {
   _isFinished = true;
   sprintf(_terminationReason, "Maximal condition number %7.2e reached. maxEW=%7.2e, minEig=%7.2e, maxdiagC=%7.2e, mindiagC=%7.2e\n",
                                _termCondCovCond, maxEW, minEW, maxdiagC, mindiagC);
 }

 double fac;
 size_t iAchse = 0;
 size_t iKoo = 0;
 if( _isTermCondMinStepFac )
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
      _isFinished = true;
      sprintf(_terminationReason, "Standard deviation %f*%7.2e in principal axis %ld without effect.", _termCondMinStepFac, sigma*axisD[iAchse], iAchse);
      break;
    }
  }
 }

 /* Component of rgxmean is not changed anymore */
 if( _isTermCondMinStepFac )
 for (iKoo = 0; iKoo < _k->N; ++iKoo)
 {
  if (rgxmean[iKoo] == rgxmean[iKoo] + _termCondMinStepFac*sigma*sqrt(C[iKoo][iKoo]) ) //TODO: standard dev add to _isTermCond..
  {
   /* C[iKoo][iKoo] *= (1 + _covarianceMatrixLearningRate); */
   /* flg = 1; */
   _isFinished = true;
   sprintf(_terminationReason, "Standard deviation %f*%7.2e in coordinate %ld without effect.", _termCondMinStepFac, sigma*sqrt(C[iKoo][iKoo]), iKoo);
   break;
  }

 } /* for iKoo */

 if( _isTermCondMaxFitnessEvaluations && (countevals >= _termCondMaxFitnessEvaluations) )
 {
  _isFinished = true;
  sprintf(_terminationReason, "Conducted %lu function evaluations >= (%lu).", countevals, _termCondMaxFitnessEvaluations);
 }

 if( _isTermCondMaxGenerations && (_currentGeneration >= _termCondMaxGenerations) )
 {
  _isFinished = true;
  sprintf(_terminationReason, "Maximum number of Generations reached (%lu).", _termCondMaxGenerations);
 }

 return _isFinished;
}


void CMAES::updateEigensystem(double **M, int flgforce)
{
 if(flgforce == 0 && flgEigensysIsUptodate) return;
 /* if(_currentGeneration % _covarianceEigenEvalFreq == 0) return; */

 eigen(_k->N, M, axisDtmp, Btmp);
 
 /* find largest and smallest eigenvalue, they are supposed to be sorted anyway */
 double minEWtmp = doubleRangeMin(axisDtmp, _k->N);
 double maxEWtmp = doubleRangeMax(axisDtmp, _k->N);

 if (minEWtmp <= 0.0) { if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: Min Eigenvalue smaller or equal 0.0 (%+6.3e) after Eigen decomp (no update possible).\n", minEWtmp ); return; }

 for (size_t d = 0; d < _k->N; ++d) 
 {
     axisDtmp[d] = sqrt(axisDtmp[d]); 
     if (std::isfinite(axisDtmp[d]) == false)
     {
       if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: Could not calculate root of Eigenvalue (%+6.3e) after Eigen decomp (no update possible).\n", axisDtmp[d] ); 
       return; 
     }
    for (size_t e = 0; e < _k->N; ++e) if (std::isfinite(B[d][e]) == false)
    {
       if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: Non finite value detected in B (no update possible).\n"); 
       return;
    }
 }
 
 /* write back */
 for (size_t d = 0; d < _k->N; ++d) axisD[d] = axisDtmp[d];
 for (size_t d = 0; d < _k->N; ++d) memcpy(B[d], Btmp[d], sizeof(double) * _k->N );

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

 gsl_eigen_symmv (&m.matrix, gsl_eval, gsl_evec, gsl_work);
 gsl_eigen_symmv_sort (gsl_eval, gsl_evec, GSL_EIGEN_SORT_ABS_ASC);

 for (size_t i = 0; i < size; i++)
 {
  gsl_vector_view gsl_evec_i = gsl_matrix_column (gsl_evec, i);
  for (size_t j = 0; j < size; j++) Q[j][i] = gsl_vector_get (&gsl_evec_i.vector, j);
 }

 for (size_t i = 0; i < size; i++) diag[i] = gsl_vector_get (gsl_eval, i);

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
 if (_isFinished || (_currentGeneration % _resultOutputFrequency) == 0) _k->saveState(_currentGeneration);
}


void CMAES::printGeneration() const
{
 if (_currentGeneration % _resultOutputFrequency != 0) return;

 if (_k->_verbosity >= KORALI_MINIMAL)
   printf("[Korali] Generation %ld - Duration: %fs (Total Elapsed Time: %fs)\n", _currentGeneration, std::chrono::duration<double>(t1-t0).count(), std::chrono::duration<double>(t1-startTime).count());

 if ( _hasConstraints && _isViabilityRegime && _k->_verbosity >= KORALI_NORMAL)
 {
   printf("\n[Korali] CCMA-ES searching start (MeanX violates constraints) .. \n");
   printf("[Korali] Viability Bounds:\n");
   for (size_t c = 0; c < _k->_fconstraints.size(); c++)  printf("         %s = (%+6.3e)\n", _k->_variables[c]->_name.c_str(), viabilityBounds[c]);
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

  printf("[Korali] Constraint Evaluation at Current Function Value:\n");
  if ( _hasConstraints )
  if ( bestValidIdx >= 0 )
  {
      for (size_t c = 0; c < _k->_fconstraints.size(); c++) printf("         ( %+6.3e )\n", constraintEvaluations[c][bestValidIdx]);
  }
  else
  {
      for (size_t c = 0; c < _k->_fconstraints.size(); c++) printf("         ( %+6.3e )\n", constraintEvaluations[c][0]);
  }

  printf("[Korali] Covariance Matrix:\n");
  for (size_t d = 0; d < _k->N; d++)
  {
   for (size_t e = 0; e <= d; e++) printf("   %+6.3e  ",C[d][e]);
   printf("\n");
  }

  printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
  printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
  if ( _hasConstraints ) { printf("[Korali] Number of Constraint Evaluations: %zu\n", countcevals); printf("[Korali] Number of Matrix Corrections: %zu\n", correctionsC ); }
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
    for (size_t d = 0; d < _k->N; ++d) printf("         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), rgxbestever[d]);
    if ( _hasConstraints ) { printf("[Korali] Constraint Evaluation at Optimum:\n"); for (size_t c = 0; c < _k->_fconstraints.size(); c++) printf("         ( %+6.3e )\n", besteverCeval[c]); }
    printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
    printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
    printf("[Korali] Stopping Criterium: %s\n", _terminationReason);
    printf("[Korali] Total Elapsed Time: %fs\n", std::chrono::duration<double>(endTime-startTime).count());
    printf("--------------------------------------------------------------------\n");
 }
}
