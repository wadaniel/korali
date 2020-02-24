#!/usr/bin/env python3

import sys
import os
import json
import shutil
import copy

toolsSrcDir = '../../../tools'
toolsDstDir = '../using/tools'

shutil.rmtree(toolsDstDir, ignore_errors=True, onerror=None)
os.makedirs(toolsDstDir)

shutil.copyfile(toolsSrcDir + '/plotter/README.rst', toolsDstDir + '/plotter.rst')
