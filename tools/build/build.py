import sys
import os
import glob
import argparse
from builders import *


print("\n[Korali] Start Parser")

parser = argparse.ArgumentParser()
parser.add_argument('--dir', '-d', default='./', help='Korali root directory')
args = parser.parse_args()

dir = {}
dir['root'] = os.path.abspath( args.dir )
dir['source'] = os.path.join( dir['root'], 'source' )
dir['include'] = os.path.join( dir['root'], 'include')
dir['modules'] = os.path.join( dir['source'], 'modules' )

# These header files are copied directly. No build is needed.
headerFileList = glob.glob(dir['source']+'/auxiliar/*.hpp')
headerFileList = [ './auxiliar/'+os.path.split(_file)[1] for _file in  headerFileList ]
headerFileList += [ 'engine.hpp',
                    'korali.hpp',
                    'sample/sample.hpp',
                    'modules/solver/learner/deepSupervisor/optimizers/fCMAES.hpp',
                    'modules/solver/learner/deepSupervisor/optimizers/fAdaBelief.hpp',
                    'modules/solver/learner/deepSupervisor/optimizers/fAdam.hpp',
                    'modules/module.hpp'
                  ]


builder = codeBuilder(dir)

for moduleDir, relDir, fileNames in os.walk(dir['modules']):
  for fileName in fileNames:
    builder.buildHeadersAndSource(moduleDir, fileName)

builder.buildVariableHeader()

for _file in headerFileList:
  builder.copyNoBuildHeaders(_file)

print("[Korali] End Parser\n")



