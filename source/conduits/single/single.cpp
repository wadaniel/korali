#include "korali.h"

using namespace Korali::Conduit;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Single::Single(nlohmann::json& js) : Base::Base(js)
{
 setConfiguration(js);
}

Single::~Single()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Single::getConfiguration()
{
 auto js = this->Base::getConfiguration();

 js["Ranks Per Team"] = 1;

 return js;
}

void Single::setConfiguration(nlohmann::json& js)
{
 this->Base::setConfiguration(js);

 int ranksPerTeam = consume(js, { "Ranks Per Team" }, KORALI_NUMBER, std::to_string(1));
 if (ranksPerTeam != 1)
 {
  fprintf(stderr, "[Korali] Error: You have defined 'Ranks Per Team' = %d, but did not run Korali using MPI. \n", ranksPerTeam);
  fprintf(stderr, "[Korali] Solution: Run Korali using MPI (e.g., $mpirun -n NUMBER ./myprogram)\n");
  exit(-1);
 }
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Single::run()
{
 _k->_solver->run();
}

void Single::evaluateSample(double* sampleArray, size_t sampleId)
{
 double fitness = _k->_problem->evaluateFitness(&sampleArray[_k->_problem->N*sampleId], true);
 _k->_solver->processSample(sampleId, fitness);
}

void Single::checkProgress()
{

}

bool Single::isRoot()
{
 return true;
}
