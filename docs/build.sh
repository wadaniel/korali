#!/bin/bash

function check()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error building site."
  exit -1
 fi 
}

# Installing mkdocs and materials theme
pip install mkdocs --user
check

pip install mkdocs-material --user
check

python3 -m mkdocs build
check

# Copying website to falcon (gateway)
scp -r site/* circleci@falcon.ethz.ch:websites/korali
check

# Copying from falcon to vladimirovich (host)
ssh circleci@falcon.ethz.ch './update_vladimirovich.sh'
check

echo "[Korali] Building complete."
