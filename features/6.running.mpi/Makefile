BINARIES = run-tmcmc run-cmaes
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o model/jacobi.o
	$(KORALICXX) -o $@ $^ $(KORALILIBS)

model/jacobi.o:
	$(MAKE) -C model all

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS)

.PHONY: clean
clean:
	$(MAKE) -C model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
