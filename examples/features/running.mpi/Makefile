BINARIES = run-tmcmc run-cmaes
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o _model/jacobi.o
	$(MPICXX) -o $@ $^ $(KORALILIBS)

_model/jacobi.o:
	$(MAKE) -C _model all

%.o: %.cpp
	$(MPICXX) -c $< $(KORALICFLAGS)

.PHONY: clean
clean:
	$(MAKE) -C _model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
