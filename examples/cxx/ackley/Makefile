BINARIES = maximize_function sample_function
OBJECTS = model/ackley.o
CXX = korali-cxx

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o $(OBJECTS)
	$(CXX) -o $@ $^ 

model/ackley.o: 
	$(MAKE) -C model all 

%.o: %.cpp
	$(CXX) -c $<

.PHONY: clean
clean:
	$(MAKE) -C model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
