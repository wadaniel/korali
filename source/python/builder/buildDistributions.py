import sys
import os
import json
from buildAux import *

def buildDistributions(koraliDir):
 # Processing Distributions
 curdir = koraliDir + '/distributions' 
 
 # Creating distribution creation list
 distributionCreationList = ''
 
 # Detecting Distributions
 distributionPaths  = [x[0] for x in os.walk(curdir)][1:]
 for distributionPath in distributionPaths:
  distributionName = distributionPath.replace(curdir + '/', '')
  
  # Loading JSON Configuration
  distributionJsonFile = distributionPath + '/' + distributionName + '.json'
  if (not os.path.isfile(distributionJsonFile)): continue 
  with open(distributionJsonFile, 'r') as file: distributionJsonString = file.read()
  distributionConfig = json.loads(distributionJsonString)
  
  # Creating check for conditional properties string
  conditionalCheckString = ''
 
  ####### Adding distribution to list
  
  distributionCreationList += '  if(distributionType == "' + distributionConfig["Alias"] + '") distribution = new Korali::Distribution::' + distributionConfig["Class"] + '();\n'
  
  ####### Producing distribution.hpp
  
  # Producing private variable declarations
  distributionHeaderString = ''
  
  for v in distributionConfig["Distribution Configuration"]:
   distributionHeaderString += 'double ' + getCXXVariableName(v) + ';\n'
   distributionHeaderString += 'std::string ' + getCXXVariableName(v) + 'Conditional;\n'
     
  for v in distributionConfig["Internal Settings"]:
   distributionHeaderString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
       
  # Loading template header .hpp file
  distributionTemplateHeaderFile = distributionPath + '/' + distributionName + '._hpp'
  with open(distributionTemplateHeaderFile, 'r') as file: distributionTemplateHeaderString = file.read()
  newHeaderString = distributionTemplateHeaderString.replace('private:', 'private: \n' + distributionHeaderString + '\n')
  
  # Saving new header .hpp file
  distributionNewHeaderFile = distributionPath + '/' + distributionName + '.hpp'
  print('[Korali] Creating: ' + distributionNewHeaderFile + '...')
  with open(distributionNewHeaderFile, 'w') as file: file.write(newHeaderString)
  
  ###### Producing distribution.cpp
  
  distributionCodeString = 'void Korali::Distribution::' + distributionConfig["Class"] + '::setConfiguration(nlohmann::json& js) \n{\n'
 
  # Consume Distribution Settings
  for v in distributionConfig["Distribution Configuration"]:
    distributionCodeString += ' ' + getCXXVariableName(v) + 'Conditional = "";\n'
    distributionCodeString += ' if(js' + getVariablePath(v) + '.is_number()) ' + getCXXVariableName(v) + ' = js' + getVariablePath(v) + ';\n'
    distributionCodeString += ' if(js' + getVariablePath(v) + '.is_string()) ' + getCXXVariableName(v) + 'Conditional = js' + getVariablePath(v) + ';\n' 
    distributionCodeString += ' eraseValue(js, "' + getVariablePath(v).replace('"', "'") + '");\n\n' 
    
  for v in distributionConfig["Internal Settings"]:
    distributionCodeString += consumeValue('_k->_js', distributionConfig["Alias"], '["Internal"]' + getVariablePath(v),  getCXXVariableName(v), getVariableType(v), 'Korali Skip Default')
  
  distributionCodeString += '} \n\n'
  
  ###### Creating Distribution Get Configuration routine
  
  distributionCodeString += 'void Korali::Distribution::' + distributionConfig["Class"]  + '::getConfiguration(nlohmann::json& js) \n{\n\n'
  distributionCodeString += 'js["Type"] = "' + distributionConfig["Alias"] + '";\n'
 
  for v in distributionConfig["Distribution Configuration"]: 
    distributionCodeString += ' if(' + getCXXVariableName(v) + 'Conditional == "") js' + getVariablePath(v) + ' = ' + getCXXVariableName(v) + ';\n'
    distributionCodeString += ' if(' + getCXXVariableName(v) + 'Conditional != "") js' + getVariablePath(v) + ' = ' + getCXXVariableName(v) + 'Conditional;\n'
    
  for v in distributionConfig["Internal Settings"]: 
    distributionCodeString += 'js["Internal"]' + getVariablePath(v) + ' = ' + getCXXVariableName(v) + ';\n'
  
  distributionCodeString += '} \n\n'
  
  ###### Creating Set property routine
  
  distributionCodeString += 'void Korali::Distribution::' + distributionConfig["Class"]  + '::setProperty(std::string propertyName, double value)\n'
  distributionCodeString += '{\n'
  distributionCodeString += ' bool recognizedProperty = false;\n'
 
  for v in distributionConfig["Distribution Configuration"]: 
    distributionCodeString += ' if (propertyName == "' + v["Name"][-1] + '") { ' + getCXXVariableName(v) + ' = value; recognizedProperty = true; }\n'
  
  distributionCodeString += ' if (recognizedProperty == false) koraliError("Unrecognized property: %s for the ' + distributionConfig["Alias"] + ' distribution", propertyName);\n'
  distributionCodeString += '} \n\n'
  
  ###### Creating Check for conditional properties
  
  for v in distributionConfig["Distribution Configuration"]: 
    conditionalCheckString += ' bool ' + getCXXVariableName(v) + 'Recognized = false;\n'
    
  conditionalCheckString += ' for (size_t i = 0; i < _k->N; i++)\n'
  conditionalCheckString += ' {\n'
    
  for v in distributionConfig["Distribution Configuration"]: 
    conditionalCheckString += '  if (_k->_variables[i]->_name == ' + getCXXVariableName(v) + 'Conditional) \n'
    conditionalCheckString += '  {\n'
    conditionalCheckString += '   ' + getCXXVariableName(v) + 'Recognized = true;\n'
    conditionalCheckString += '   hasConditionals = true;\n'
    
    conditionalCheckString += '   _conditionalIndexes.push_back(i);\n'
    conditionalCheckString += '   _conditionalPointers.push_back(&' + getCXXVariableName(v) + ');\n'
    conditionalCheckString += '  }\n'
    
  conditionalCheckString += ' }\n'

  for v in distributionConfig["Distribution Configuration"]: 
    conditionalCheckString += ' if ( !' + getCXXVariableName(v) + 'Conditional.empty() && ' + getCXXVariableName(v) + 'Recognized == false)\n'
    conditionalCheckString += '  koraliError("Could not find a variable whose name coincides with reference %s made by conditional property: ' + v["Name"][-1] + '.\\n", ' + getCXXVariableName(v) + 'Conditional);\n'
  
  ###### Creating code file
  
  with open(distributionPath + '/' + distributionName + '._cpp', 'r') as file: distributionBaseCodeString = file.read()
  distributionBaseCodeString += '\n\n' + distributionCodeString
  distributionBaseCodeString = distributionBaseCodeString.replace(' // Check for conditional properties', conditionalCheckString)
  distributionNewCodeFile = distributionPath + '/' + distributionName + '.cpp'
  print('[Korali] Creating: ' + distributionNewCodeFile + '...')
  with open(distributionNewCodeFile, 'w') as file: file.write(distributionBaseCodeString)
 
 ###### Creating base configuration file
 with open(curdir + '/base._cpp', 'r') as file: distributionBaseCodeString = file.read()
 newBaseString = distributionBaseCodeString.replace(' // Distribution list', distributionCreationList) 
 with open(curdir + '/base.cpp', 'w+') as file: file.write(newBaseString)
 
 ###### Creating base header file
 with open(curdir + '/base._hpp', 'r') as file: distributionBaseHeaderString = file.read()
 newBaseString = distributionBaseHeaderString
 with open(curdir + '/base.hpp', 'w+') as file: file.write(newBaseString)
 
