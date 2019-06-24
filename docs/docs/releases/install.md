# Installation

## Steps

### 1. Check your system

 Check the [Requirements](../requirements) page to make sure your system provides
 the required compilation and software requirements for Korali.

### 2. Download Korali
Open a command line console and download and extract the latest release of Korali.

```shell
wget 'https://www.cse-lab.ethz.ch/korali/releases/korali-latest.tar.gz'
tar -xzvf korali-latest.tar.gz 
```

### 3. Install Korali

To build and install Korali, simply run:

```shell
cd korali 
./install
```

!!! note
	Use ```./install --no-install-gsl``` if you do not with Korali to automatically download and build pre-requisites.