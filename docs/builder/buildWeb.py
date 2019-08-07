#!/usr/bin/env python3

import os
import json
import sys

koraliDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../..') 

from buildProblems import *
from buildSolvers import *
from buildModels import *
from buildTests import *
from buildConduits import *
from buildTutorials import *

with open(koraliDir + '/docs/docs/manual._md', 'r') as file: manualString = file.read()

# Processing Problems
problemListString = buildProblems(koraliDir)
   
# Processing Solvers
solverListString = buildSolvers(koraliDir)

# Processing Conduits
conduitsListString = buildConduits(koraliDir)

# Processing Models
modelsListString = buildModels(koraliDir)

# Processing Tutorials
buildTutorials(koraliDir)

# Processing Tests
buildTests(koraliDir)

newManualFileName = koraliDir + '/docs/docs/manual.md'
manualString = manualString.replace('<!--- Problems go here --->', problemListString)
manualString = manualString.replace('<!--- Solvers go here --->', solverListString)
manualString = manualString.replace('<!--- Conduits go here --->', conduitsListString)
manualString = manualString.replace('<!--- Models go here --->', modelsListString)
print('[Korali] Creating ' + newManualFileName + '...')  
with open(newManualFileName, 'w+') as file: file.write(manualString)