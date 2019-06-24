# Installation

## Steps

### 1. Check your system

 Check the [Requirements](../requirements) page to make sure your system provides
 the required compilation and software requirements for Korali.

### 2. Download Korali

Download the latest release of Korali from the following path:

+ [**korali-latest.tar.gz**](https://gitlab.ethz.ch/mavt-cse/korali-releases/raw/master/korali-latest.tar.gz)

!!! attention
	You need to be logged into tour MAVT/CSE-Lab account to be able to download the file.
	
### Extract Korali
Open a command line console and extract the downloaded file

```shell
tar -xzvf korali-latest.tar.gz 
```

### 3. Install Korali

To build and install Korali, simply run:

```shell
cd korali 
./install
```

!!! note
	This step may take several minutes since Korali will automatically download and build its prerequisites.
	
	Use ```./install --no-install-gsl``` if you do not with Korali to automatically build pre-requisites.
	
### 4. Test Korali

Run any of the included tutorials to verify Korali has installed correctly.

```shell
cd tutorials
./a1-optimization
```