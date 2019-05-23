BINARIES = maximize_function sample_function
KORALICXX = korali-cxx

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o model/ackley.o
	$(KORALICXX) -o $@ $^

model/ackley.o:
	$(MAKE) -C model all

%.o: %.cpp
	$(KORALICXX) -c $<

.PHONY: clean
clean:
	$(MAKE) -C model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
