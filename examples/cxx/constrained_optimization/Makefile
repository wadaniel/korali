BINARIES = maximize_constrained_rosenbrock maximize_constrained_g09
KORALICXX = korali-cxx

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o models/models.a
	$(KORALICXX) -o $@ $^

models/models.a:
	$(MAKE) -C models all

%.o: %.cpp
	$(KORALICXX) -c $< `gsl-config --cflags`

.PHONY: clean
clean:
	$(MAKE) -C models clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
