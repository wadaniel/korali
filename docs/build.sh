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

echo "[Korali] Building complete."
