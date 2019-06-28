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
ssh circleci@falcon.ethz.ch './update_vladimirovich.sh'
check

echo "[Korali] Deploying complete."
