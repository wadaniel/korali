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

git branch | grep \* | grep master
if [ $? -eq 0 ] 
then
  echo "[Korali] Deploying to master webpage."
  ssh circleci@falcon.ethz.ch './update_master.sh'
  check
else
  echo "[Korali] Deploying to development webpage."
  ssh circleci@falcon.ethz.ch './update_development.sh'
  check
fi

echo "[Korali] Deploying complete."
