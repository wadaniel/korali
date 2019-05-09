# Installation

## Steps

### 1. Check your system

 Check the [Requirements](./requirements.md) section to make sure your system provides
 the required compilation and software requirements for Korali.

### 2. Download Korali
Download the Korali project from [GitHub](https://github.com/cselab/sKorali)

```shell
git clone https://github.com/cselab/sKorali
```

Altenatively, you can download the latest stable Korali version from:

```
wget https://cse-lab.ethz.ch/Korali/korali-latest.tar.gz
tar -xzvf Korali-latest.tar.gz
cd Korali-latest
```

### 3. Run the configuration script

Check the full syntax for [configure](./configuration.md) script here.

```shell
./configure
```
By default the install directory is `$PWD/Korali`. If you want to install in a different directory, run
```shell
./configure --prefix=/path/to/install/folder
```

### 4. Build and Install Korali

To build and install Korali, run:

```shell
make install
```

For a faster build, you can use parallel jobs:

```shell
make -j install
```

!!! note
    If not provided by the user, the GSL-2.5 GNU Scientific Library will be automatically downloaded and installed in the installation folder. This step may take a few minutes.

### 5. Add Path and Python Path folders

Finally, Korali will ask you to add its default binary and Python paths:

```shell
export PATH=$PATH:/path/to/install/folder/bin
export PYTHONPATH=$PYTHONPATH:/path/to/install/folder/lib
```

Where /path/to/install/folder will be replaced by the install folder you chose during configuration.

### 6. Run tests [Optional]

To make sure Korali runs correctly in your system, you may run:

```shell
make test
```

This command will perform a series of tests to survey the correct execution of Korali's features.
