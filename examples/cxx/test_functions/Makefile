BINARIES = maximize_function sample_function
KORALICXX = korali-cxx

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o models/models.a
	$(KORALICXX) -o $@ $^

models/models.a:
	$(MAKE) -C models all

%.o: %.cpp
	$(KORALICXX) -c $<

.PHONY: clean
clean:
	$(MAKE) -C models clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
