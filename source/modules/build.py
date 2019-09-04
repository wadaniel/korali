import sys
import os
import json

####################################################################
# Helper Functions
####################################################################

def getVariableType(v):
 # Replacing bools with ints for Python compatibility
 return v['Type'].replace('bool', 'int').replace('std::function<void(Korali::Model&)>', 'size_t')
 
def getCXXVariableName(v):
 cVarName = ''
 for name in v: cVarName += name 
 cVarName = cVarName.replace(" ", "")
 cVarName = cVarName.replace("(", "")
 cVarName = cVarName.replace(")", "")
 cVarName = cVarName.replace("+", "")
 cVarName = cVarName.replace("-", "")
 cVarName = cVarName.replace("[", "")
 cVarName = cVarName.replace("]", "")
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
 
 if ('Korali::Sample' in varType):
  cString = ''
  return cString
 
 if ('std::function' in varType):
  cString += ' ' + varName + ' = ' + base + path + '.get<size_t>();\n'
  cString += '   eraseValue(' + base + ', "' + path.replace('"', "'") + '");\n'
  return cString 
  
 if ('std::vector<Korali::' in varType):
  baseType = varType.replace('std::vector<', '').replace('>','')
  cString += ' for(size_t i = 0; i < ' + base + path + '.size(); i++) ' + varName + '.push_back((' + baseType + ')Korali::Base::getModule(' + base + path + '[i]));\n'
  cString += ' eraseValue(' + base + ', "' + path.replace('"', "'") + '");\n\n' 
  return cString
  
 if ('Korali::' in varType):
  if (varDefault): cString = ' if (! isDefined(' + base + ', "' + path.replace('"', "'") + '[\'Type\']")) ' + base + path + '["Type"] = "' + varDefault + '"; \n'
  cString += ' ' + varName + ' = dynamic_cast<' + varType + '>(Korali::Base::getModule(' + base + path + '));\n'
  cString += ' ' + varName + '->setConfiguration(' + base + path + ');\n'
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

 if ('Korali::Sample' in varType):
  sString = ''
  return sString
  
 if ('std::vector<Korali::' in varType):
  sString = ' for(size_t i = 0; i < ' + varName + '.size(); i++) ' + varName + '[i]->getConfiguration(' + base + path + '[i]);\n'
  return sString
    
 if ('Korali::' in varType):
  sString =  ' ' + varName + '->getConfiguration(' + base + path + ');\n'  
  return sString
    
 sString = '   ' + base + path + ' = ' + varName + ';\n'  
 return sString

####################################################################
 
def getParentClass(module):
  className = module["C++ Class"]
  parentName = className.rsplit('::', 1)[0]
  if ('::Base' in className): parentName = className.rsplit('::', 2)[0]
  parentName += '::Base'
  return parentName
 
####################################################################

def createSetConfiguration(module):
 codeString = 'void ' + module["C++ Class"] + '::setConfiguration(nlohmann::json& js) \n{\n'
  
 # Consume Configuration Settings
 if 'Configuration Settings' in module:
  for v in module["Configuration Settings"]:
   codeString += consumeValue('js', module["Alias"], getVariablePath(v), getCXXVariableName(v["Name"]), getVariableType(v), getVariableDefault(v))
 
 if 'Internal Settings' in module: 
  for v in module["Internal Settings"]:
   codeString += consumeValue('js', module["Alias"], '["Internal"]' + getVariablePath(v),  getCXXVariableName(v["Name"]), getVariableType(v), 'Korali Skip Default')
  
 if 'Termination Criteria' in module:
  for v in module["Termination Criteria"]:
   codeString += consumeValue('js', module["Alias"], '["Termination Criteria"]' + getVariablePath(v), getCXXVariableName(v["Name"]), getVariableType(v), getVariableDefault(v))
 
 if 'Variables Configuration' in module:
  codeString += ' for (size_t i = 0; i < _variables.size(); i++) { \n'
  for v in module["Variables Configuration"]:
   codeString += consumeValue('js["Variables"][i]', module["Alias"], getVariablePath(v), '_variables[i]->' + getCXXVariableName(v["Name"]), getVariableType(v), getVariableDefault(v))
  codeString += ' } \n'
   
 if 'Conditional Variables' in module:
  for v in module["Conditional Variables"]:
   codeString += ' ' + getCXXVariableName(v["Name"]) + 'Conditional = "";\n'
   codeString += ' if(js' + getVariablePath(v) + '.is_number()) ' + getCXXVariableName(v["Name"]) + ' = js' + getVariablePath(v) + ';\n'
   codeString += ' if(js' + getVariablePath(v) + '.is_string()) ' + getCXXVariableName(v["Name"]) + 'Conditional = js' + getVariablePath(v) + ';\n'
   codeString += ' eraseValue(js, "' + getVariablePath(v).replace('"', "'") + '");\n\n'
 
 codeString += ' ' + getParentClass(module) + '::setConfiguration(js);\n'
 
 codeString += ' if(isEmpty(js) == false) Korali::logError("Unrecognized settings for ' + module["Name"] + ' (' + module["Alias"] + '): \\n%s\\n", js.dump(2).c_str());\n'
 codeString += '} \n\n'
  
 return codeString
  
####################################################################
  
def createGetConfiguration(module):  
 codeString = 'void ' + module["C++ Class"]  + '::getConfiguration(nlohmann::json& js) \n{\n\n'
 
 if 'Configuration Settings' in module:
  for v in module["Configuration Settings"]: 
   codeString += saveValue('js', getVariablePath(v), getCXXVariableName(v["Name"]), getVariableType(v))
 
 if 'Termination Criteria' in module:
  for v in module["Termination Criteria"]: 
   codeString += saveValue('js', getVariablePath(v), getCXXVariableName(v["Name"]), getVariableType(v))
   
 if 'Internal Settings' in module:   
  for v in module["Internal Settings"]: 
   codeString += saveValue('js', '["Internal"]' + getVariablePath(v), getCXXVariableName(v["Name"]), getVariableType(v))
   
 if 'Variables Configuration' in module:
  codeString += ' for (size_t i = 0; i < _variables.size(); i++) { \n'
  for v in module["Variables Configuration"]:
   codeString += saveValue('js["Variables"][i]', getVariablePath(v), '_variables[i]->' + getCXXVariableName(v["Name"]), getVariableType(v))
  codeString += ' } \n'  
   
 if 'Conditional Variables' in module: 
  for v in module["Conditional Variables"]:
   codeString += ' if(' + getCXXVariableName(v["Name"]) + 'Conditional == "") js' + getVariablePath(v) + ' = ' + getCXXVariableName(v["Name"]) + ';\n'
   codeString += ' if(' + getCXXVariableName(v["Name"]) + 'Conditional != "") js' + getVariablePath(v) + ' = ' + getCXXVariableName(v["Name"]) + 'Conditional;\n'
 
 codeString += ' ' + getParentClass(module) + '::getConfiguration(js);\n'
 
 codeString += '} \n\n'
 
 return codeString

####################################################################

def createCheckTermination(module):  
 codeString = 'bool ' + module["C++ Class"]  + '::checkTermination()\n'
 codeString += '{\n'
 codeString += ' bool hasFinished = false;\n\n'
 
 if 'Termination Criteria' in module:
  for v in module["Termination Criteria"]: 
   codeString += ' if (' + v["Criteria"] + ')\n'
   codeString += ' {\n'
   codeString += '  Korali::logInfo("Minimal", "' + module["Alias"] + ' Termination Criteria met: \\"' + getVariablePath(v).replace('"', "'") + '\\" (' + getVariableDescriptor(v) + ').\\n", ' + getCXXVariableName(v["Name"])  +');\n'
   codeString += '  hasFinished = true;\n'
   codeString += ' }\n\n'
 
   codeString += ' hasFinished = hasFinished || ' + getParentClass(module) + '::checkTermination();\n' 
 codeString += ' return hasFinished;\n'
 codeString += '}'
 
 return codeString
 
####################################################################

def createHeaderDeclarations(module):  
 headerString = ''
 
 if 'Configuration Settings' in module:
  for v in module["Configuration Settings"]:
   headerString += ' ' + getVariableType(v) + ' ' + getCXXVariableName(v["Name"]) + ';\n'
 
 if 'Internal Settings' in module:    
  for v in module["Internal Settings"]:
   headerString += ' ' + getVariableType(v) + ' ' + getCXXVariableName(v["Name"]) + ';\n'
 
 if 'Termination Criteria' in module:
  for v in module["Termination Criteria"]:
   headerString += ' ' + getVariableType(v) + ' ' + getCXXVariableName(v["Name"]) + ';\n'
   
 if 'Conditional Variables' in module:
  for v in module["Conditional Variables"]:
   headerString += ' double ' + getCXXVariableName(v["Name"]) + ';\n'
   headerString += ' std::string ' + getCXXVariableName(v["Name"]) + 'Conditional;\n'
 
 return headerString
 
  
####################################################################

def createVariableDeclarations(module):  
 variableDeclarationString = ''
 
 if 'Variables Configuration' in module:
  for v in module["Variables Configuration"]:
   variableDeclarationString += '  ' + getVariableType(v) + ' ' + getCXXVariableName(v["Name"]) + ';\n'
 
 return variableDeclarationString
 
####################################################################
# Main Parser Routine
####################################################################

currentDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
koraliDir = currentDir + '/../' 

# modules List
moduleDetectionList = ''
moduleIncludeList = ''

# Variable Declaration List
varDeclarationSet = set()

# Detecting modules' json file
for moduleDir, relDir, fileNames in os.walk(currentDir):
 for fileName in fileNames: 
  if '.json' in fileName:
   with open(moduleDir + '/' + fileName, 'r') as file: moduleConfig = json.load(file)
   moduleName = fileName.replace('.json', '')
   
   ####### Adding module to list
   if (not '(Base)' in moduleConfig["Name"]):
    relpath = os.path.relpath(moduleDir, koraliDir)
    filepath = os.path.join(relpath, moduleName + '.hpp')
    moduleIncludeList += '#include "' + filepath + '" \n'
    moduleDetectionList += '  if(moduleType == "' + moduleConfig["Alias"] + '") module = new ' + moduleConfig["C++ Class"] + '();\n'
   
   ###### Producing module code

   getTypeString = 'std::string ' + moduleConfig["C++ Class"] + '::getType() { return "' + moduleConfig["Alias"] + '"; }'   
   moduleCodeString = createSetConfiguration(moduleConfig)
   moduleCodeString += createGetConfiguration(moduleConfig)
   moduleCodeString += createCheckTermination(moduleConfig)
   moduleCodeString += '\n\n' + getTypeString + '\n\n'
 
   ####### Producing header file
   
   # Loading template header .hpp file
   moduleTemplateHeaderFile = moduleDir + '/' + moduleName + '._hpp'
   with open(moduleTemplateHeaderFile, 'r') as file: moduleTemplateHeaderString = file.read()
   
   # Adding overridden function declarations
   functionOverrideString = ''
   functionOverrideString += ' std::string getType() override;\n'
   functionOverrideString += ' bool checkTermination() override;\n'
   functionOverrideString += ' void getConfiguration(nlohmann::json& js) override;\n'
   functionOverrideString += ' void setConfiguration(nlohmann::json& js) override;\n'
   newHeaderString = moduleTemplateHeaderString.replace('public:', 'public: \n' + functionOverrideString + '\n')
   
   # Adding declarations
   declarationsString = createHeaderDeclarations(moduleConfig)
   newHeaderString = newHeaderString.replace('public:', 'public: \n' + declarationsString + '\n')
   
   # Retrieving variable declarations
   for varDecl in createVariableDeclarations(moduleConfig).splitlines():
    varDeclarationSet.add(varDecl)
   
   # Saving new header .hpp file
   moduleNewHeaderFile = moduleDir + '/' + moduleName + '.hpp'
   print('[Korali] Creating: ' + moduleNewHeaderFile + '...')
   with open(moduleNewHeaderFile, 'w') as file: file.write(newHeaderString)
   
   ###### Creating code file
   
   moduleBaseCodeFileName = moduleDir + '/' + moduleName + '._cpp'
   moduleNewCodeFile = moduleDir + '/' + moduleName + '.cpp'
   baseFileTime = os.path.getmtime(moduleBaseCodeFileName)
   newFileTime = baseFileTime
   if (os.path.exists(moduleNewCodeFile)): newFileTime = os.path.getmtime(moduleNewCodeFile)
   
   if (baseFileTime >= newFileTime):
    with open(moduleBaseCodeFileName, 'r') as file: moduleBaseCodeString = file.read()
    moduleBaseCodeString += '\n\n' + moduleCodeString
    print('[Korali] Creating: ' + moduleNewCodeFile + '...')
    with open(moduleNewCodeFile, 'w') as file: file.write(moduleBaseCodeString)

###### Updating module source file 

moduleBaseCodeFileName = currentDir + '/base._cpp' 
moduleNewCodeFile = currentDir + '/base.cpp'
baseFileTime = os.path.getmtime(moduleBaseCodeFileName)
newFileTime = baseFileTime
if (os.path.exists(moduleNewCodeFile)): newFileTime = os.path.getmtime(moduleNewCodeFile)

if (baseFileTime >= newFileTime):
  with open(moduleBaseCodeFileName, 'r') as file: moduleBaseCodeString = file.read()
  newBaseString = moduleBaseCodeString.replace('// Module Include List',  moduleIncludeList)
  newBaseString = newBaseString.replace(' // Module Selection List', moduleDetectionList)
  print('[Korali] Creating: ' + moduleNewCodeFile + '...')
  with open(moduleNewCodeFile, 'w') as file: file.write(newBaseString)

###### Updating base header file

moduleBaseHeaderFileName = currentDir + '/base._hpp'
moduleNewHeaderFile = currentDir + '/base.hpp'
with open(moduleBaseHeaderFileName, 'r') as file: moduleBaseHeaderString = file.read()
newBaseString = moduleBaseHeaderString
print('[Korali] Creating: ' + moduleNewHeaderFile + '...')
with open(moduleNewHeaderFile, 'w+') as file: file.write(newBaseString)

###### Updating variable header file
variableDeclarationList = ''
for varDecl in varDeclarationSet:
 variableDeclarationList += varDecl + '\n'

variableBaseHeaderFileName = currentDir + '/variable/variable.hpp'
variableNewHeaderFile = currentDir + '/variable/variable.hpp'
with open(variableBaseHeaderFileName, 'r') as file: variableBaseHeaderString = file.read()
newBaseString = variableBaseHeaderString
newBaseString = newBaseString.replace(' // Variable Declaration List', variableDeclarationList)
print('[Korali] Creating: ' + variableNewHeaderFile + '...')
with open(variableNewHeaderFile, 'w+') as file: file.write(newBaseString)
