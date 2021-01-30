#!/usr/bin/env python3

class Args:
    def __init__(self, input, config=None, output=None):
      self.input = input
      self.config = config
      self.output = output


from build import *

configFileList = []
headerFileList = []
sourceFileList = []
modulesDir = '../../source/modules'
for moduleDir, relDir, fileNames in os.walk( modulesDir ):
    for fileName in fileNames:
        if '.config' in fileName:
            _name = os.path.splitext(fileName)[0]
            configFileList.append(os.path.abspath(os.path.join(moduleDir,_name+'.config')))
            headerFileList.append(os.path.abspath(os.path.join(moduleDir,_name+'._hpp')))
            sourceFileList.append(os.path.abspath(os.path.join(moduleDir,_name+'._cpp')))

for configFile, headerFile, sourceFile in zip(configFileList, headerFileList, sourceFileList):
    args = Args( [headerFile,sourceFile], configFile )
    main(args)


from build_variables_header import *

variableHeaderTemplateFile = '../../source/variable/variable._hpp'
configFileList = []
modulesDir = '../../source/modules/'
for moduleDir, relDir, fileNames in os.walk( modulesDir ):
    for fileName in fileNames:
        if '.config' in fileName:
            _name = os.path.splitext(fileName)[0]
            configFileList.append(os.path.join(moduleDir,_name+'.config'))

args = Args( [variableHeaderTemplateFile] + configFileList )

main(args)