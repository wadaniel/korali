#!/bin/bash

function check()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error building site."
  exit -1
 fi 
}



# Installing shpinx, mkdocs, and materials theme
python3 -m pip install sphinx --user
check

python3 -m pip install sphinx_rtd_theme --user
check

python3 -m pip install mkdocs --user
check

python3 -m pip install mkdocs-material --user
check

# Building main website
pushd website

python3 -m mkdocs build
check

# Some re-styling changes
sed -i -e 's/76\./36\./g;s/12\.1/10\.5/g' ./site/assets/stylesheets/application.*.css
check

popd 

# Building User Manual
pushd manual

make html
check

popd

# Inserting user manual into website

cp -r manual/.build/html website/site/docs
check

echo "[Korali] Webpage Build complete."
