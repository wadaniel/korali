INCLUDES = $(shell cd source && find . | grep "\.h")
TESTS = $(dir $(wildcard tests/*/))
CURDIR = $(shell pwd)
PIP ?= python3 -m pip

include .korali.config

.PHONY: all install clean snapshot tests clean_tests $(TESTS)

KORALI_LIBNAME_SHARED=source/libkorali.so

all: $(KORALI_LIBNAME_SHARED)

$(KORALI_LIBNAME_SHARED):
	@mkdir -p $(PREFIX)/lib
	@$(MAKE) -j -C source

clean: 
	@$(MAKE) -j -C source clean
	@rm -f setup.py

install: $(KORALI_LIBNAME_SHARED) 
	@echo "[Korali] Installing Korali..."
	@mkdir -p $(PREFIX)
	@mkdir -p $(PREFIX)/lib
	@mkdir -p $(PREFIX)/include
	@mkdir -p $(PREFIX)/bin
	@cp $(KORALI_LIBNAME_SHARED) $(PREFIX)/lib
	@ln -sf $(KORALI_LIBNAME_SHARED) $(PREFIX)/lib/libkorali.dylib
	@cd source && for i in $(INCLUDES); do rsync -R $$i $(PREFIX)/include > /dev/null 2>&1; done 
	@echo "#!/bin/bash" > $(PREFIX)/bin/korali-cxx
	@cat .korali.config source/auxiliar/bin/korali-cxx >> $(PREFIX)/bin/korali-cxx
	@chmod a+x  $(PREFIX)/bin/korali-cxx
	@ln -sf ./source/auxiliar/python/setup/setup.py setup.py
	@$(PIP) install . --user --upgrade
	@rm -f setup.py
	@echo '------------------------------------------------------------------'
	@echo '[Korali] Finished installation successfully.'
	@echo '[Korali] Do not forget to update your PATH environment:'
	@echo '[Korali] >export PATH=$$PATH:$(PREFIX)/bin'
	@echo '------------------------------------------------------------------' 

snapshot: install clean
	tar -zcvf korali`date +"%m-%d-%y"`.tar.gz korali/ tests/
