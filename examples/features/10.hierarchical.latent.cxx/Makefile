BINARIES =  run-saem-hierarchical   run-saem-hierarchical-n-d   test_hsaem   estimate-FIM-hierarchical-n-d   estimate-FIM-hierarchical-n-d-4-jobs estimate-FIM-hierarchical-n-d-8-jobs

baseKoraliDir=../..
include $(baseKoraliDir)/source/Makefile.conf
# externalDir=${baseKoraliDir}/external
# GSLDir=${externalDir}/GSL/install/include


KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=$(shell python3 -m korali.cxx --cflags)
KORALILIBS=$(shell python3 -m korali.cxx --libs)
KORALICFLAGS += $(GSLCFLAGS)
KORALILIBS += $(GSLLIBS)

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o _model/model.o  _model/load_data.o   _model/utils.o
	$(KORALICXX) -g -o $@ $^ $(KORALILIBS)

_model/%.o:
	$(MAKE) -C _model all

%.o: %.cpp
	$(KORALICXX) -g -c $(KORALICFLAGS) $<

.PHONY: clean
clean:
	$(MAKE) -C _model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
