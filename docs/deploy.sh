#!/bin/bash

function check()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error deploying site."
  exit -1
 fi 
}

currentBranch=`git rev-parse --abbrev-ref HEAD`
if [[ $currentBranch !=  "master" ]]; then
 echo "This is not Korali's master branch so webpage deployment is not performed."
 exit 0
fi

pushd website

# Copying website to falcon (gateway)
echo $FALCON_FINGERPRINT >> ~/.ssh/known_hosts
tar -zcvf site.tar.gz site
scp -r site.tar.gz circleci@falcon.ethz.ch:websites/korali
check

# Copying from falcon to vladimirovich (host)

echo "[Korali] Deploying to webpage."
ssh circleci@falcon.ethz.ch './update.sh'
check

popd

echo "[Korali] Deploying complete."
