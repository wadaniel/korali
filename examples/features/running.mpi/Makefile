BINARIES = run-tmcmc run-cmaes
CXX=mpic++
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o _model/jacobi.o
	$(CXX) -o $@ $^ $(KORALILIBS)

_model/jacobi.o:
	$(MAKE) -C _model all

%.o: %.cpp
	$(CXX) -c $< $(KORALICFLAGS)

.PHONY: clean
clean:
	$(MAKE) -C _model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
