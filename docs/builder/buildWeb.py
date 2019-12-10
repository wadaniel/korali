#!/usr/bin/env python3

import os
import json
import sys

koraliDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../..') 

from buildModules import *
from buildTests import *
from buildTutorials import *

# Processing Modules
buildModules(koraliDir)

# Processing Tutorials
buildTutorials(koraliDir)

# Processing Tests
buildTests(koraliDir)

with open(koraliDir + '/docs/docs/manual/manual._md', 'r') as file: manualString = file.read()
newManualFileName = koraliDir + '/docs/docs/manual/manual.md'
print('[Korali] Creating ' + newManualFileName + '...')  
with open(newManualFileName, 'w+') as file: file.write(manualString)
