import sys
import os
import json

#####################################################################

def getVariableType(v):
 # Replacing bools with ints for Python compatibility
 return v['Type'].replace('bool', 'int')
 
def getCXXVariableName(v):
 cVarName = ''
 for name in v["Name"]: cVarName += name 
 cVarName = cVarName.replace(" ", "")
 cVarName = '_' + cVarName[0].lower() + cVarName[1:]
 return cVarName

def getVariablePath(v):
 cVarPath = ''
 for name in v["Name"]: cVarPath += '["' + name + '"]' 
 return cVarPath
 
def getVariableDefault(v):
 return v.get('Default', '')
 
def getVariableEnabledDefault(v):
 if ( v.get('Default', '') ): return 'true'
 return 'false'
 
def getVariableDescriptor(v):
 if ('size_t' in v['Type']): return '%lu'
 if ('int' in v['Type']): return '%d'
 if ('bool' in v['Type']): return '%b'
 if ('double' in v['Type']): return '%e'
 if ('float' in v['Type']): return '%e'
 print('Error: Unrecognized type')
 exit(-1)
 
#####################################################################

def consumeValue(base, moduleName, path, varName, varType, varDefault):
 cString = '\n'
 
 if (varType == 'Korali::Distribution::Base*'):
  cString =  ' ' + varName + ' = Korali::Distribution::Base::getDistribution(_k->_problem, ' + base + path + ');\n'
  cString += ' eraseValue(' + base + ', "' + path.replace('"', "'") + '");\n' 
  return cString
  
 if (varType == 'std::vector<Korali::Distribution::Base*>'):
  cString  = ' for(size_t i = 0; i < ' + base + path + '.size(); i++)' + varName + '.push_back(Korali::Distribution::Base::getDistribution(this, ' + base + path + '[i]));\n'
  cString += ' eraseValue(' + base + ', "' + path.replace('"', "'") + '");\n\n' 
  return cString
 
 if (varType == 'std::vector<Korali::Variable*>'):
  cString  = ' for(size_t i = 0; i < ' + base + path + '.size(); i++)' + varName + '.push_back(new Korali::Variable(this)); \n'
  cString += ' for(size_t i = 0; i < ' + base + path + '.size(); i++)' + varName + '[i]->setConfiguration(' + base + path + '[i]);\n'
  cString += ' eraseValue(' + base + ', "' + path.replace('"', "'") + '");\n\n' 
  return cString
 
 cString += ' if (isDefined(' + base + ', "' + path.replace('"', "'") + '"))  \n  { \n'
 cString += '   ' + varName + ' = ' + base + path + '.get<' + varType + '>();\n' 
 cString += '   eraseValue(' + base + ', "' + path.replace('"', "'") + '");\n'
 cString += '  }\n'
 
 if (varDefault == 'Korali Skip Default'):
  return cString
 
 cString += '  else '
 if (varDefault == ''):
  cString += '  Korali::logError("No value provided for mandatory setting: ' + path.replace('"', "'") + ' required by ' + moduleName + '.\\n"); \n'
 else:
  if ("std::string" in varType): varDefault = '"' + varDefault + '"'
  cString += varName + ' = ' + varDefault + ';'
   
 cString += '\n'
 return cString

#####################################################################

def saveValue(base, path, varName, varType):
 sString = '   ' + base + path + ' = ' + varName + ';\n'

 if (varType == 'Korali::Distribution::Base*'):
  sString =  ' ' + varName + '->getConfiguration(' + base + path + ');\n'
  
 if (varType == 'std::vector<Korali::Distribution::Base*>'):
  sString  = ' for(size_t i = 0; i < ' + varName + '.size(); i++) ' + varName + '[i]->getConfiguration(' + base + path + '[i]);\n'
  
 if (varType == 'std::vector<Korali::Variable*>'):
  sString  = ' for(size_t i = 0; i < ' + varName + '.size(); i++) ' + varName + '[i]->getConfiguration(' + base + path + '[i]);\n'
 
 return sString
 
####################################################################

def createSetConfiguration(module):
 codeString = 'void Korali::' + module["Type"] + '::' + module["C++ Class"] + '::setConfiguration(nlohmann::json& js) \n{\n'
  
 # Erase type, if exists.
 codeString += ' if(isDefined(js, "[\'Type\']")) eraseValue(js, "[\'Type\']");\n'

 # Checking whether solver is accepted
 if 'Compatible Solvers' in module:
  codeString += ' bool __acceptedSolver = false;\n'
  for v in module["Compatible Solvers"]: 
   codeString += ' if (_k->_solverType == "' + v + '") __acceptedSolver = true;\n'
  codeString += ' if (__acceptedSolver == false) Korali::logError("Selected solver %s not compatible with  type ' + module["Name"] + '", _k->_solverType.c_str()); \n\n' 
 
 # Consume Configuration Settings
 if 'Configuration Settings' in module:
  for v in module["Configuration Settings"]:
   codeString += consumeValue('js', module["Alias"], getVariablePath(v), getCXXVariableName(v), getVariableType(v), getVariableDefault(v))
 
 if 'Internal Settings' in module: 
  for v in module["Internal Settings"]:
   codeString += consumeValue('js', module["Alias"], '["Internal"]' + getVariablePath(v),  getCXXVariableName(v), getVariableType(v), 'Korali Skip Default')
  
 if 'Termination Criteria' in module:
  for v in module["Termination Criteria"]:
   codeString += consumeValue('js', module["Alias"], '["Termination Criteria"]' + getVariablePath(v), getCXXVariableName(v), getVariableType(v), getVariableDefault(v))
 
 codeString += ' if(isEmpty(js) == false) Korali::logError("Unrecognized settings for ' + module["Name"] + ' (' + module["Alias"] + '): \\n%s\\n", js.dump(2).c_str());\n'
 codeString += '} \n\n'
  
 return codeString
  
####################################################################
  
def createGetConfiguration(module):  
 codeString = 'void Korali::' + module["Type"] + '::' + module["C++ Class"]  + '::getConfiguration(nlohmann::json& js) \n{\n\n'
 
 for v in module["Configuration Settings"]: 
  codeString += saveValue('js', getVariablePath(v), getCXXVariableName(v), getVariableType(v))
    
 for v in module["Internal Settings"]: 
  codeString += saveValue('js', '["Internal"]' + getVariablePath(v), getCXXVariableName(v), getVariableType(v))
  
 codeString += '} \n\n'
 
 return codeString

####################################################################

def createCheckTermination(module):  
 codeString = 'bool Korali::' + module["Type"] + '::' + module["C++ Class"]  + '::checkTermination()\n'
 codeString += '{\n'
 codeString += ' bool hasFinished = false;\n\n'
 
 if 'Termination Criteria' in module:
  for v in module["Termination Criteria"]: 
   codeString += ' if (' + v["Criteria"] + ')\n'
   codeString += ' {\n'
   codeString += '  Korali::logInfo("Minimal", "' + module["Alias"] + ' Termination Criteria met: \\"' + getVariablePath(v).replace('"', "'") + '\\" (' + getVariableDescriptor(v) + ').\\n", ' + getCXXVariableName(v)  +');\n'
   codeString += '  hasFinished = true;\n'
   codeString += ' }\n\n'
  
 codeString += ' return hasFinished;\n'
 codeString += '}'
 
 return codeString
  