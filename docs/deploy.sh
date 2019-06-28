#!/bin/bash

function check()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error deploying site."
  exit -1
 fi 
}

# Copying website to falcon (gateway)
echo $FALCON_FINGERPRINT >> ~/.ssh/known_hosts
scp -r site/* circleci@falcon.ethz.ch:websites/korali
check

# Copying from falcon to vladimirovich (host)

git branch | grep master
if [ $? -eq 0 ] 
then
  ssh circleci@falcon.ethz.ch './update_master.sh'
  check
else
  ssh circleci@falcon.ethz.ch './update_development.sh'
  check
fi

echo "[Korali] Deploying complete."
