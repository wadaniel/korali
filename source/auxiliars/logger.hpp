#ifndef _KORALI_AUXILIARS_LOGGER_HPP_
#define _KORALI_AUXILIARS_LOGGER_HPP_

namespace Korali
{

enum _korali_verbosity_levels_ { KORALI_UNDEFINED = -1, KORALI_SILENT = 0, KORALI_MINIMAL = 1, KORALI_NORMAL = 2, KORALI_DETAILED = 3 };

extern _korali_verbosity_levels_ _korali_verbosity;

void logData(const _korali_verbosity_levels_ level, const char* format, ... );
void logInfo(const _korali_verbosity_levels_ level, const char* format, ... );
void logWarning(const _korali_verbosity_levels_ level, const char* format, ... );
void logError(const char* format, ... );

}
#endif
