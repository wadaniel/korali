BINARIES = maximize_function sample_function
OBJECTS = models/ackley.o models/eggholder.o models/levi13.o models/rosenbrock.o
CXX = korali-cxx

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o $(OBJECTS)
	$(CXX) -o $@ $^ 

models/%.o:
	$(MAKE) -C models all

%.o: %.cpp
	$(CXX) -c $<

.PHONY: clean
clean:
	$(MAKE) -C models clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
