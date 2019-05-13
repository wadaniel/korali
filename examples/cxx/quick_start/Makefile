BINARIES = direct_optimize direct_sample posterior_optimize posterior_sample 
OBJECTS =
CXX = korali-cxx

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o $(OBJECTS)
	$(CXX) -o $@ $^

%.o: %.cpp
	$(CXX) -c $<

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
