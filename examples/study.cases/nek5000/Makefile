BINARIES = run-korali
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

NEKDIR=_deps/nek5000
NEK5000CFLAGS=
NEK5000LIBS=$(NEKDIR)/examples/turbChannel/turbChannel.o $(NEKDIR)/examples/turbChannel/libnek5000.a -L$(NEKDIR)/3rd_party/blasLapack -lblasLapack -L$(NEKDIR)/3rd_party/gslib/lib -lgs -Wl,--allow-multiple-definition -lgfortran

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o environment.o
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(NEK5000LIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(NEK5000CFLAGS)

environment.o: _environment/environment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(NEK5000CFLAGS)
 
.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.h5 *.dat *.xmf
