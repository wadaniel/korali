INCLUDES = $(shell cd source && find . | grep "\.h")
TESTS = $(dir $(wildcard tests/*/))

.PHONY: all install clean snapshot tests clean_tests $(TESTS)
include common.mk

all: libkorali.so

libkorali.a:
	@$(MAKE) -j -C source

clean: clean_tests
	@$(MAKE) -j -C source clean

tests: $(TESTS)

$(TESTS):: install
	$(MAKE) -j -C $@

clean_tests:
	for i in $(TESTS); do $(MAKE) -j -C $$i clean; done

install: libkorali.a
	@echo "[Korali] Installing Korali..."
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	mkdir -p $(PREFIX)/bin
	cp source/libkorali.so $(PREFIX)/lib
	@cd source && cp $(INCLUDES) --parents $(PREFIX)/include
	cp libs/json -r $(PREFIX)/include
	cp libs/koralijson -r $(PREFIX)/include
	@cat tools/korali-cxx | sed -e 's%(LDFLAGS)%$(LDFLAGS)%g' -e 's%(CXXFLAGS)%$(CXXFLAGS)%g' -e 's%(CXX)%$(CXX)%g' -e 's%(PREFIX)%$(PREFIX)%g' > $(PREFIX)/bin/korali-cxx
	@chmod a+x  $(PREFIX)/bin/korali-cxx

snapshot: install clean
	tar -zcvf korali`date +"%m-%d-%y"`.tar.gz korali/ tests/
