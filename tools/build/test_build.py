#!/usr/bin/env python3
from pathlib import Path
import codeBuilders.auxiliar as aux
import os

class Args:
    def __init__(self, input, config=None, output=None):
      self.input = input
      self.config = config
      self.output = output


def makeTestPath( codeFilePath ):
  p = Path( codeFilePath )
  p = p.resolve()
  r = aux.pathSplitAtDir( p, 'source' )
  r = p.relative_to(r)
  t = Path('./test_source/').joinpath(r)
  suffix = t.suffix.replace('_','')
  t = t.with_suffix(suffix)
  return str(t)


from build import *

configFileList = []
headerFileList = []
sourceFileList = []
outHeaderFileList = []
outSourceFileList = []
modulesDir = '../../source/modules'
for moduleDir, relDir, fileNames in os.walk( modulesDir ):
    for fileName in fileNames:
        if '.config' in fileName:
            _name = os.path.splitext(fileName)[0]
            configFileList.append(os.path.abspath(os.path.join(moduleDir,_name+'.config')))

            _file = os.path.abspath(os.path.join(moduleDir,_name+'._hpp'))
            headerFileList.append(_file)
            outHeaderFileList.append( makeTestPath(_file) )

            _file = os.path.abspath(os.path.join(moduleDir,_name+'._cpp'))
            sourceFileList.append(_file)
            outSourceFileList.append( makeTestPath(_file) )

for config, inHeader, inSource, outHeader, outSource in \
  zip(configFileList, headerFileList, sourceFileList, outHeaderFileList, outSourceFileList):
    print( [inHeader,inSource], config, [outHeader,outSource] )
    args = Args( [inHeader,inSource], config, [outHeader,outSource] )
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

outHeader = makeTestPath(variableHeaderTemplateFile)
args = Args( [variableHeaderTemplateFile] + configFileList, output=outHeader )

main(args)