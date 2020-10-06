BINARIES = run-sdpg
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
MSODECFLAGS=-I_deps/msode/build/src/msode/core -I_deps/msode/build/src/msode/rl -I_deps/msode/build/src/msode/analytic_control -I_deps/msode/build/src/msode/utils
MSODELIBS=_deps/msode/build/src/msode/core/libmsode.a _deps/msode/build/src/msode/rl/librl.a  _deps/msode/build/src/msode/analytic_control/libanalytic_control.a _deps/msode/build/src/msode/utils/libutils.a

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o model.o libcubism.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(MSODELIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(MSODECFLAGS)

model.o: _model/model.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(MSODECFLAGS)

.PHONY: clean
clean:
	$(MAKE) -C _model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.h5 *.dat *.xmf
