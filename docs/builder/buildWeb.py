#!/usr/bin/env python3

import os
import json
import sys

koraliDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../..') 

from buildSolvers import *
from buildProblems import *
from buildTutorials import *

mkdocsFileSrc = koraliDir + '/docs/mkdocs._yml'
with open(mkdocsFileSrc, 'r') as file: mkdocsSourceString = file.read()
 
# Processing Solvers
solverYamlEntries = buildSolvers(koraliDir)
mkdocsSourceString = mkdocsSourceString.replace('- Solvers:', '- Solvers:\n' + solverYamlEntries)

# Processing Problems
problemYamlEntries = buildProblems(koraliDir)
mkdocsSourceString = mkdocsSourceString.replace('- Problems:', '- Problems:\n' + problemYamlEntries)

# Processing Tutorials
problemYamlEntries = buildTutorials(koraliDir)
mkdocsSourceString = mkdocsSourceString.replace('- Tutorials:', '- Tutorials:\n' + problemYamlEntries)

mkdocsFileDst = koraliDir + '/docs/mkdocs.yml'
print('[Korali] Creating ' + mkdocsFileDst + '...')
with open(mkdocsFileDst, 'w+') as file: file.write(mkdocsSourceString)