BINARIES = direct posterior
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
