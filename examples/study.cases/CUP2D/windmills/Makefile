WINDMILL = run-vracer-windmill eval-vracer-windmill 
WINDMILL4 = 4run-vracer-windmill 4eval-vracer-windmill
KORALICXX=${CXX}
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
CUPFOLDER=../_deps/CUP-2D
CUBISMCFLAGS=$(shell cat ${CUPFOLDER}/makefiles/cup.cflags.txt) -I${CUPFOLDER}/source -I${CUPFOLDER}/Cubism/include -DCUBISM -Wno-shadow
CUBISMLIBS= ${CUPFOLDER}/makefiles/libcup.a $(shell cat ${CUPFOLDER}/makefiles/cup.libs.txt) -lstdc++fs

.SECONDARY:
.PHONY: all
all: windmill #4windmill

windmill: $(WINDMILL)
#4windmill: $(WINDMILL4)

$(WINDMILL): %: %.o windmillEnvironment.o ${CUPFOLDER}/makefiles/libcup.a 
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

#$(WINDMILL4): %: %.o 4windmillEnvironment.o ${CUPFOLDER}/makefiles/libcup.a 
#	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

windmillEnvironment.o: _model/windmillEnvironment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

#4windmillEnvironment.o: _model/4windmillEnvironment.cpp
#	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

.PHONY: clean
clean:
	$(RM) $(WINDMILL) *.o *.ti *.optrpt *.h5 *.xmf
