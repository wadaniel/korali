INCLUDES = $(shell cd source && find . | grep "\.h")
TESTS = $(dir $(wildcard tests/*/))
CURDIR = $(shell pwd)

include korali.config

.PHONY: all install clean snapshot tests clean_tests $(TESTS)

KORALI_LIBNAME_SHARED=source/libkorali.so
KORALI_LIBNAME_STATIC=source/libkorali.a

GSL_LIBNAME=libs/gsl/lib/libgsl.$(DLL_EXTENSION)

all: $(KORALI_LIBNAME_SHARED)

$(KORALI_LIBNAME_SHARED): $(GSL_LIBNAME)
	@$(MAKE) -j -C source

clean: 
	@$(MAKE) -j -C source clean

tests: $(TESTS)

$(TESTS):: install
	$(MAKE) -j -C $@

clean_tests:
	for i in $(TESTS); do $(MAKE) -j -C $$i clean; done

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
	cp -r libs/gsl/include/gsl $(PREFIX)/include 
	cp libs/gsl/lib/libgsl* $(PREFIX)/lib/
	cp tools/korali-plot $(PREFIX)/bin
	chmod a+x $(PREFIX)/bin/korali-plot
	cp -r tools/plotting/* $(PREFIX)/bin/
	chmod a+x  $(PREFIX)/bin/plot_cmaes.py
	chmod a+x  $(PREFIX)/bin/plot_tmcmc.py
	@echo "#!/bin/bash" > $(PREFIX)/bin/korali-cxx
	@cat korali.config tools/korali-cxx >> $(PREFIX)/bin/korali-cxx
	@chmod a+x  $(PREFIX)/bin/korali-cxx
	@echo '------------------------------------------------------------------'
	@echo '[Korali] To complete installation, please update your environment:'
	@echo '[Korali] >export PATH=$$PATH:$(PREFIX)/bin'
	@echo '[Korali] >export PYTHONPATH=$$PYTHONPATH:$(PREFIX)/lib'
	@echo '------------------------------------------------------------------' 

snapshot: install clean
	tar -zcvf korali`date +"%m-%d-%y"`.tar.gz korali/ tests/

$(GSL_LIBNAME):
	@echo "[Korali] Downloading GNU Scientific Library... "
	@cd libs/ && rm -f gsl-2.5.tar.gz && wget "ftp://ftp.gnu.org/gnu/gsl/gsl-2.5.tar.gz" && tar -xzvf gsl-2.5.tar.gz > /dev/null 2>&1
	@echo "[Korali] Configuring GNU Scientific Library... "
	@cd libs/gsl-2.5 && ./configure --prefix=$(CURDIR)/libs/gsl > /dev/null 2>&1 
	@echo "[Korali] Compiling GNU Scientific Library... "
	@cd libs/gsl-2.5 && make > /dev/null 2>&1 && make -j8 install > /dev/null 2>&1
	@rm -rf libs/gsl-2.5 libs/gsl-2.5.tar.gz > /dev/null 2>&1
