BINARIES = maximize_function sample_function
OBJECTS = model/rosenbrock.o
KCXX = korali-cxx

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o $(OBJECTS)
	$(KCXX) -o $@ $^

model/%.o: 
	$(MAKE) -C model all 

%.o: %.cpp
	$(KCXX) -c $<

.PHONY: clean
clean:
	$(MAKE) -C model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
