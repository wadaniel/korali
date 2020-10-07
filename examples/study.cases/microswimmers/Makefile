BINARIES = run-sdpg
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
MSODECFLAGS=-I_deps/msode/src -I../_deps/msode/src -I_deps/msode/extern/json/include/
MSODELIBS=_deps/msode/build/lib/libgtest.a \
          _deps/msode/build/src/msode/rl/librl.a\
          -lpthread \
          _deps/msode/build/src/msode/analytic_control/libanalytic_control.a\
          _deps/msode/build/src/msode/utils/libutils.a \
          _deps/msode/build/src/msode/core/libmsode.a \
          -lbfd -ldl

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o model.o
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(MSODELIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(MSODECFLAGS)

model.o: _model/model.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(MSODECFLAGS)

.PHONY: clean
clean:
	$(MAKE) -C _model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.h5 *.dat *.xmf
