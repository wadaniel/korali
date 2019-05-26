INCLUDES = $(shell cd source && find . | grep "\.h")
TESTS = $(dir $(wildcard tests/*/))
CURDIR = $(shell pwd)
PIP ?= python3 -m pip

include .korali.config

.PHONY: all install clean snapshot tests clean_tests $(TESTS)

KORALI_LIBNAME_SHARED=source/libkorali.so
KORALI_LIBNAME_STATIC=source/libkorali.a

all: $(KORALI_LIBNAME_SHARED)

$(KORALI_LIBNAME_SHARED):
	@$(MAKE) -j -C source

clean: 
	@$(MAKE) -j -C source clean
	@rm -f setup.py

install: $(KORALI_LIBNAME_SHARED) 
	@echo "[Korali] Installing Korali..."
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	mkdir -p $(PREFIX)/bin
	mkdir -p $(PREFIX)/python
	cp $(KORALI_LIBNAME_SHARED) $(PREFIX)/lib
	ln -sf $(KORALI_LIBNAME_SHARED) $(PREFIX)/lib/libkorali.dylib
	cp $(KORALI_LIBNAME_STATIC) $(PREFIX)/lib
	cd source && for i in $(INCLUDES); do rsync -R $$i $(PREFIX)/include > /dev/null 2>&1; done 
	cp -r libs/json $(PREFIX)/include
	cp -r libs/koralijson $(PREFIX)/include
	cp tools/korali-plot $(PREFIX)/bin
	chmod a+x $(PREFIX)/bin/korali-plot
	cp -r tools/plotting/* $(PREFIX)/bin/
	chmod a+x  $(PREFIX)/bin/plot_cmaes.py
	chmod a+x  $(PREFIX)/bin/plot_tmcmc.py
	@echo "#!/bin/bash" > $(PREFIX)/bin/korali-cxx
	@cat .korali.config tools/korali-cxx >> $(PREFIX)/bin/korali-cxx
	@chmod a+x  $(PREFIX)/bin/korali-cxx
	@echo "from libkorali import *" > source/__init__.py 
	@ln -sf ./tools/setup.py setup.py
	@$(PIP) install . --user --upgrade
	@echo '------------------------------------------------------------------'
	@echo '[Korali] To finalize installation, please update your environment:'
	@echo '[Korali] >export PATH=$$PATH:$(PREFIX)/bin'
	@echo '------------------------------------------------------------------' 

snapshot: install clean
	tar -zcvf korali`date +"%m-%d-%y"`.tar.gz korali/ tests/
