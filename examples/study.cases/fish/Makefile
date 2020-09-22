BINARIES = run-ddpg
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
CUBISMCFLAGS=`cat cubism.cflags.txt` -I_deps/cubism/source -I_deps/cubism/Cubism/include
CUBISMLIBS=`cat cubism.libs.txt`

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o model.o libcubism.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

model.o: _model/model.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

.PHONY: clean
clean:
	$(MAKE) -C _model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt *.h5 *.dat *.xmf
