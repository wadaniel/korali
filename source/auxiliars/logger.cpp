#include <stdlib.h>
#include <stdio.h>
#include "logger.hpp"
#include "korali.hpp"

size_t Korali::getVerbosityLevel(std::string verbosityLevel)
{
 if (verbosityLevel == "Silent") return 0;
 if (verbosityLevel == "Minimal") return 1;
 if (verbosityLevel == "Normal") return 2;
 if (verbosityLevel == "Detailed") return 3;
}

bool Korali::isEnoughVerbosity(std::string verbosityLevel)
{
  size_t messageLevel = getVerbosityLevel(verbosityLevel);
  size_t koraliLevel = getVerbosityLevel(_k->_consoleOutputVerbosity);

  if (messageLevel <= koraliLevel) return true;
  return false;
}

void Korali::logData(std::string verbosityLevel, const char* format, ... )
{
 if (Korali::_k->_conduit != nullptr) if(! Korali::_k->_conduit->isRoot()) return;
 if (isEnoughVerbosity(verbosityLevel) == false) return;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, format, ap);

 fprintf(stdout, "%s", outstr);
 fflush(stdout);
 free(outstr);

}

void Korali::logInfo(std::string verbosityLevel, const char* format, ... )
{
 if (Korali::_k->_conduit != nullptr) if(! Korali::_k->_conduit->isRoot()) return;
 if (isEnoughVerbosity(verbosityLevel) == false) return;

 std::string newFormat = "[Korali] ";
 newFormat += format;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, newFormat.c_str(), ap);

 fprintf(stdout, "%s", outstr);
 fflush(stdout);
 free(outstr);
}

void Korali::logWarning(std::string verbosityLevel, const char* format, ... )
{
 if (Korali::_k->_conduit != nullptr) if(! Korali::_k->_conduit->isRoot()) return;
 if (isEnoughVerbosity(verbosityLevel) == false) return;

 std::string newFormat = "[Korali] Warning: ";
 newFormat += format;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, newFormat.c_str(), ap);

 fprintf(stderr, "%s", outstr);
 fflush(stderr);
 free(outstr);
}

void Korali::logError(const char* format, ... )
{
 std::string newFormat = "[Korali] Error: ";
 newFormat += format;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, newFormat.c_str(), ap);

 fprintf(stderr, "%s", outstr);

 fflush(stderr);
 free(outstr);

 std::abort();
}
