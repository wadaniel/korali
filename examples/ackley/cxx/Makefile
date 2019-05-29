BINARIES = maximize_function maximize_function_restart sample_function
KORALICXX = korali-cxx

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o model/ackley.o
	$(KORALICXX) -o $@ $^

model/ackley.o:
	$(MAKE) -C model all

%.o: %.cpp
	$(KORALICXX) -c $< `gsl-config --cflags` 

.PHONY: clean
clean:
	$(MAKE) -C model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
