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
tar -zcvf site.tar.gz site
scp -r site.tar.gz circleci@falcon.ethz.ch:websites/korali
check

# Copying from falcon to vladimirovich (host)

echo "[Korali] Deploying to webpage."
ssh circleci@falcon.ethz.ch './update.sh'
check

echo "[Korali] Deploying complete."
