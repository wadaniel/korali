import sys
import json
from pathlib import Path

from . import header_builders as hb
from . import source_builders as sb
from . import auxiliar as aux
from . import variables as vr


def buildHeaderString( configFilePath, templateFilePath ):

  moduleTemplateString = templateFilePath.read_text()

  hb.checkHeaderTemplateString( templateFilePath, moduleTemplateString )

  moduleConfig = aux.loadModuleConfiguration( configFilePath, moduleTemplateString )

  headerString = hb.createHeaderDoxygenString(moduleConfig)

  headerString += moduleTemplateString

  overrideFunctionString = hb.createOverrideFunctionString(moduleConfig)
  headerString = headerString.replace('public:', 'public: \n' + overrideFunctionString + '\n')

  calssDoxygenString = hb.createrClassDoxygenString( moduleConfig )
  headerString = headerString.replace('class ', calssDoxygenString + 'class ')

  declarationsString = hb.createHeaderDeclarations(moduleConfig)
  headerString = headerString.replace('public:', 'public: \n' + declarationsString + '\n')

  headerString = aux.replaceKeys( moduleConfig, headerString );

  return headerString



def buildCodeString( configFilePath, templateFilePath ):

  moduleTemplateString = templateFilePath.read_text()

  sb.checkSourceTemplateString( templateFilePath, moduleTemplateString )

  moduleConfig = aux.loadModuleConfiguration( configFilePath, moduleTemplateString )

  sourceString = sb.createSetConfiguration(moduleConfig)
  sourceString += sb.createGetConfiguration(moduleConfig)
  sourceString += sb.createApplyModuleDefaults(moduleConfig)
  sourceString += sb.createApplyVariableDefaults(moduleConfig)

  if 'Termination Criteria' in moduleConfig:
    sourceString += sb.createCheckTermination(moduleConfig)

  if 'Available Operations' in moduleConfig:
    sourceString += sb.createRunOperation(moduleConfig)

  if 'Conditional Variables' in moduleConfig:
    sourceString += sb.createGetPropertyPointer(moduleConfig)

  sourceString = moduleTemplateString.replace( '@endNamespace',  sourceString + '\n\n@endNamespace')

  sourceString = aux.replaceKeys( moduleConfig, sourceString );

  return sourceString




def buildCodeFromTemplate( configFile, templateFile, outputFile=None ):
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

  if outputFile == None:
    suffix = templateFilePath.suffix.replace('_','')
    filePath = templateFilePath.with_suffix(suffix)
  else:
    filePath = Path(outputFile)

  filePath.write_text(codeString)



def buildVariablesHeader(configFileList, templateFile, outputFile=None ):
  variableDeclarationList = ''
  variableDeclarationSet = set()

  for _file in configFileList:
    p = Path(_file)
    p = p.resolve()

    moduleConfig = json.loads( p.read_text() )
    string = ''

    if 'Variables Configuration' in moduleConfig:
      for v in moduleConfig["Variables Configuration"]:
        varName = vr.getCXXVariableName(v["Name"])
        if (not varName in variableDeclarationSet):
          string += '/**\n'
          string += '* @brief [Module: ' + moduleConfig["Module Data"]["Class Name"] + '] ' + v["Description"] + '\n'
          string += '*/\n'
          string += '  ' + vr.getVariableType(v) + ' ' + varName + ';\n'
          variableDeclarationSet.add(varName)

    variableDeclarationList += string

  variableHeaderTemplatePath = Path(templateFile)
  variableHeaderTemplateString = variableHeaderTemplatePath.read_text()
  variableHeaderString = variableHeaderTemplateString.replace('// Variable Declaration List', variableDeclarationList)

  if outputFile == None:
    filePath = variableHeaderTemplatePath.with_suffix('.hpp')
  else:
    filePath = Path(outputFile)

  filePath.write_text(variableHeaderString)
