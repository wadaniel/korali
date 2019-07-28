#!/usr/bin/env python3

import os
import json
import sys

koraliDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../..') 

from buildSolvers import *

mkdocsFileSrc = koraliDir + '/docs/mkdocs._yml'
with open(mkdocsFileSrc, 'r') as file: mkdocsSourceString = file.read()
 
# Processing Solvers
solverYamlEntries = buildSolvers(koraliDir)
mkdocsSourceString = mkdocsSourceString.replace('- Solvers:', '- Solvers:\n' + solverYamlEntries)

mkdocsFileDst = koraliDir + '/docs/mkdocs.yml'
print('[Korali] Creating ' + mkdocsFileDst + '...')
with open(mkdocsFileDst, 'w+') as file: file.write(mkdocsSourceString)