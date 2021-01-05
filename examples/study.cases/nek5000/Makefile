BINARIES = run-korali
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

AGENTLIB=libagent.so
NEKDIR=_deps/nek5000
NEK5000LIBS=-shared -L$(NEKDIR)/3rd_party/blasLapack -lblasLapack -L$(NEKDIR)/3rd_party/gslib/lib -lgs -Wl,--allow-multiple-definition -lgfortran
ENVLIBOBJECTS=$(NEKDIR)/examples/turbChannel/obj/*.o $(NEKDIR)/examples/turbChannel/turbChannel.o agent.o

.SECONDARY:
.PHONY: all 
all: $(BINARIES) $(AGENTLIB)

$(BINARIES) : % : %.o environment.o 
	$(KORALICXX) -o $@ $^ $(KORALILIBS) 

agent.o: _environment/agent.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) -fPIC
	
environment.o: _environment/environment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) 

libagent.so: $(ENVLIBOBJECTS)
	$(KORALICXX) -o $@ $^ $(NEK5000LIBS) 
 
.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.so *.ti *.optrpt *.h5 *.dat *.xmf
