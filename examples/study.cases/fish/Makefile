BINARIES = run-korali
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
BACKWARDLIBS = -lbfd -ldl
ifeq ($(TEST), true)
 CUBISMCFLAGS=-DTEST
 else
 CUBISMCFLAGS=$(shell cat cubism.cflags.txt) -I_deps/cubism/source -I_deps/cubism/Cubism/include -DCUBISM
 CUBISMLIBS= libcubism.a $(shell cat cubism.libs.txt) -lstdc++fs
endif

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o environment.o backward.o
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS) $(BACKWARDLIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

environment.o: _environment/environment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

backward.o: _deps/backward-cpp/backward.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)


.PHONY: clean
clean:
	$(MAKE) -C _environment clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.h5 *.dat *.xmf
