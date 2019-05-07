BINARIES = minimize_function sample_function
OBJECTS = model/eggholder.o
CXX = korali-cxx

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o $(OBJECTS)
	$(CXX) -o $@ $^

model/%.o: 
	$(MAKE) -C model all 

%.o: %.cpp
	$(CXX) -c $<

.PHONY: clean
clean:
	$(MAKE) -C model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
