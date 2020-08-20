BINARIES =  run-saem-hierarchical   run-saem-hierarchical-n-d   test_hsaem   estimate-FIM-hierarchical-n-d
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

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
