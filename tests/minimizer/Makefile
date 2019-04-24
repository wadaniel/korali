BINARIES = eval_minimizer
CXX = korali-cxx

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) -c $<

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
