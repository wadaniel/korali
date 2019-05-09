# Installation

## Steps

### 1. Check your system

 Check the [Requirements](./requirements.md) section to make sure your system provides
 the required compilation and software requirements for Korali.

### 2. Download Korali
Download the Korali project from [GitHub](https://github.com/cselab/skorali)

```shell
git clone https://github.com/cselab/skorali
```

tar -xzvf

### 3. Run the configuration script

Check the full syntax for [configure](./configuration.md) script here.

```shell
./configure
```
By default the install directory is `./korali`. If you want to install in a different directory, run
```shell
./configure --prefix=/path/to/your/directory
```

### 4. Build and Install Korali

Finally, install Korali
```shell
make install
```
In the beginning the GSL-2.5 GNU Scientific Library will be downloaded and installed in your system. This will take some time.

### 5. Add Path and Python Path folders

**Don't forget to add the binary and library folders to the path!**

### 6. Run tests [Optional]

To make sure Korali runs correctly.
