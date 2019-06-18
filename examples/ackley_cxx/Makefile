BINARIES = maximize_function_de maximize_function_cmaes maximize_function_restart sample_function_tmcmc sample_function_mcmc
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o model/ackley.o
	$(KORALICXX) -o $@ $^  $(KORALILIBS)

model/ackley.o:
	$(MAKE) -C model all

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS)

.PHONY: clean
clean:
	$(MAKE) -C model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
