import sys
from pathlib import Path

from header_builders import *
from source_builders import *
from auxiliar import *


def buildHeaderString( configFilePath, templateFilePath ):

  moduleTemplateString = templateFilePath.read_text()

  checkHeaderTemplateString( templateFilePath, moduleTemplateString )

  moduleConfig = loadModuleConfiguration( configFilePath, moduleTemplateString )

  headerString = createHeaderDoxygenString(moduleConfig)

  headerString += moduleTemplateString

  overrideFunctionString = createOverrideFunctionString(moduleConfig)
  headerString = headerString.replace('public:', 'public: \n' + overrideFunctionString + '\n')

  calssDoxygenString = createrClassDoxygenString( moduleConfig )
  headerString = headerString.replace('class ', calssDoxygenString + 'class ')

  declarationsString = createHeaderDeclarations(moduleConfig)
  headerString = headerString.replace('public:', 'public: \n' + declarationsString + '\n')

  headerString = replaceKeys( moduleConfig, headerString );

  return headerString



def buildCodeString( configFilePath, templateFilePath ):

  moduleTemplateString = templateFilePath.read_text()

  checkSourceTemplateString( templateFilePath, moduleTemplateString )

  moduleConfig = loadModuleConfiguration( configFilePath, moduleTemplateString )

  sourceString = createSetConfiguration(moduleConfig)
  sourceString += createGetConfiguration(moduleConfig)
  sourceString += createApplyModuleDefaults(moduleConfig)
  sourceString += createApplyVariableDefaults(moduleConfig)

  if 'Termination Criteria' in moduleConfig:
    sourceString += createCheckTermination(moduleConfig)

  if 'Available Operations' in moduleConfig:
    sourceString += createRunOperation(moduleConfig)

  if 'Conditional Variables' in moduleConfig:
    sourceString += createGetPropertyPointer(moduleConfig)

  sourceString = moduleTemplateString.replace( '@endNamespace',  sourceString + '\n\n@endNamespace')

  sourceString = replaceKeys( moduleConfig, sourceString );

  return sourceString




def buildCodeFromTemplate( configFile, templateFile   ):
  configFilePath = Path(configFile)
  configFilePath = configFilePath.resolve()

  templateFilePath = Path(templateFile)
  templateFilePath.resolve()

  if configFilePath.suffix != '.config' :
    sys.exit(f'[Korali] Error: {configFilePath} is not a .config file.\n')

  if configFilePath.parent != templateFilePath.parent:
    sys.exit('[Korali] Error: configuration file and template file are not in the same directory.')

  if '._hpp' == templateFilePath.suffix:
    codeString = buildHeaderString( configFilePath, templateFilePath )
  elif '._cpp' == templateFilePath.suffix:
    codeString = buildCodeString( configFilePath, templateFilePath )
  else:
    sys.exit('[Korali] Error: Unknown extension in template file.\n')

  suffix = templateFilePath.suffix.replace('_','')
  filePath = templateFilePath.with_suffix(suffix)
  filePath.write_text(codeString)
