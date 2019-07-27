#####################################################################

def getVariableType(v):
 # Replacing bools with ints for Python compatibility
 return v['Type'].replace('bool', 'int')
 
def getVariableName(v):
 cVarName = v["Name"].replace(" ", "")
 cVarName = '_' + cVarName[0].lower() + cVarName[1:]
 return cVarName
 
def getVariableDefault(v):
 return v.get('Default', '')

#####################################################################

def consumeValue(base, moduleName, settingName, varName, varType, varDefault, path = []):
 cString = '\n'
 
 cString += ('  if (isDefined(' + base + ', {')
 for p in path: cString += (' "' + p +'",')
 cString += (' "' + settingName + '"})) \n  { \n')

 cString += ('   ' + varName + ' = ' + base )
 for p in path: cString += ('.at("' + p +'")')
 getLine = '.at("' + settingName + '").get<' + varType + '>();\n'
 cString += getLine
 
 cString += ('   ' + base)
 for p in path: cString += ('.at("' + p +'")')
 cString += ('.erase("' + settingName + '");\n')
 
 cString += ('  }\n')
 
 if (varDefault == 'Korali Skip Default'):
  return cString
 
 cString += ('  else ')
 if (varDefault == ''):
  cString += ('  koraliError("No value provided for mandatory setting: ')
  for p in path: cString += ('[' + p + '] > ')
  cString += ('[' + settingName + '], required by ' + moduleName + '.\\n"); ')
 else:
  if ("std::string" in varType): varDefault = '"' + varDefault + '"'
  defaultLine = varName + ' = ' + varDefault + ';'
  if ('std::vector<' in varType):
   defaultLine = 'for(size_t i = 0; i < ' + varName + '.size(); i++) ' + varName + '[i] = ' + varDefault + ';'
  if ('std::vector<std::vector<' in varType):
   defaultLine = 'for(size_t i = 0; i < ' + varName + '.size(); i++) ' + 'for(size_t j = 0; j < ' + varName + '[i].size(); j++) ' + varName + '[i][j] = ' + varDefault + ';'
  cString += (defaultLine)
   
 cString += ('\n')
 return cString

#####################################################################

