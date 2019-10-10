#!/bin/bash

# this will exit the script if a command fails
set -e

cd phase0
python3 x.py
cd ../

cd phase1
sh generateData.sh
cd ../

cd phase2
python3 x.py
cd ../

cd phase3a
python3 x.py
cd ../

cd phase3b
python3 x.py
cd ../
