#!/usr/bin/env python3

import os
import json
import sys

koraliDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../..') 

from buildSolvers import *
from buildProblems import *
from buildTutorials import *

# Processing Solvers
buildSolvers(koraliDir)

# Processing Problems
buildProblems(koraliDir)

# Processing Tutorials
buildTutorials(koraliDir)

