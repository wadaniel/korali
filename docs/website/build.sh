#!/bin/bash

function check()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error building site."
  exit -1
 fi 
}

pushd website

# Build pages from source code
pushd builder
python3 ./buildWeb.py
check
popd

# Installing mkdocs and materials theme
python3 -m pip install mkdocs --user
check

python3 -m pip install mkdocs-material --user
check

python3 -m mkdocs build
check

# Some re-styling changes
sed -i -e 's/76\./36\./g;s/12\.1/10\.5/g' ./site/assets/stylesheets/application.*.css
check

popd 

echo "[Korali] Webpage Build complete."
