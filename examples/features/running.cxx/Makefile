BINARIES = run-cmaes-direct run-cmaes run-lmcma-direct run-lmcma run-tmcmc
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

ifndef MPICXX
	MPICXX=mpic++
endif

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o _model/heat2d.o
	$(MPICXX) -o $@ $^ $(KORALILIBS)

_model/%.o: 
	$(MAKE) -C _model all 

%.o: %.cpp
	$(MPICXX) -c $(KORALICFLAGS) $<

.PHONY: clean
clean:
	$(MAKE) -C _model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
