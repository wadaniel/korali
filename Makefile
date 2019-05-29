INCLUDES = $(shell cd source && find . | grep "\.h")
TESTS = $(dir $(wildcard tests/*/))
CURDIR = $(shell pwd)
PIP ?= python3 -m pip

.PHONY: all install clean snapshot tests clean_tests $(TESTS)

KORALI_LIBNAME_SHARED=libkorali.so
KORALI_LIBNAME_STATIC=libkorali.a

all: source/$(KORALI_LIBNAME_SHARED) source/$(KORALI_LIBNAME_STATIC)

source/$(KORALI_LIBNAME_SHARED):
	@$(MAKE) -j -C source $(KORALI_LIBNAME_SHARED)
	
source/$(KORALI_LIBNAME_STATIC):
	@$(MAKE) -j -C source $(KORALI_LIBNAME_STATIC)

clean: 
	@$(MAKE) -j -C source clean
	@rm -f setup.py

install: source/$(KORALI_LIBNAME_SHARED) source/$(KORALI_LIBNAME_STATIC)
	@echo "[Korali] Installing Korali..."
	@cat ./python/cxx/cxx.conf ./python/cxx/cflags.py.base > ./python/cxx/cflags.py
	@cat ./python/cxx/cxx.conf ./python/cxx/compiler.py.base > ./python/cxx/compiler.py
	@cat ./python/cxx/cxx.conf ./python/cxx/libs.py.base > ./python/cxx/libs.py
	@ln -sf ./python/setup/setup.py setup.py
	@$(PIP) install . --user --upgrade
	@rm -f setup.py
	@echo '------------------------------------------------------------------'
	@echo '[Korali] Finished installation successfully.'
	@echo '------------------------------------------------------------------' 

snapshot: install clean
	tar -zcvf korali`date +"%m-%d-%y"`.tar.gz korali/ tests/
