KORALICXX=${CXX}
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.PHONY: clean all
all: run-vracer-swimmer-2D run-vracer-swimmer-3D eval-vracer-swimmer-2D eval-vracer-swimmer-3D

#         2D          #
#######################
CUP2DFOLDER=../../CUP2D/_deps/CUP-2D
CUBISM2DCFLAGS=$(shell cat ${CUP2DFOLDER}/makefiles/cup.cflags.txt) -I${CUP2DFOLDER}/source -I${CUP2DFOLDER}/Cubism/include -DCUBISM -Wno-shadow
CUBISM2DLIBS= ${CUP2DFOLDER}/makefiles/libcup.a $(shell cat ${CUP2DFOLDER}/makefiles/cup.libs.txt) -lstdc++fs
swimmerEnvironment2D.o:
	$(KORALICXX) -c _model2D/swimmerEnvironment2D.cpp $(KORALICFLAGS) $(CUBISM2DCFLAGS)
run-vracer-swimmer-2D.o:
	$(KORALICXX) -c -o run-vracer-swimmer-2D.o run-vracer-swimmer.cpp $(KORALICFLAGS) $(CUBISM2DCFLAGS) -DmodelDIM=2
run-vracer-swimmer-2D: run-vracer-swimmer-2D.o swimmerEnvironment2D.o ${CUP2DFOLDER}/makefiles/libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISM2DLIBS)
eval-vracer-swimmer-2D.o:
	$(KORALICXX) -c -o eval-vracer-swimmer-2D.o eval-vracer-swimmer.cpp $(KORALICFLAGS) $(CUBISM2DCFLAGS) -DmodelDIM=2
eval-vracer-swimmer-2D: eval-vracer-swimmer-2D.o swimmerEnvironment2D.o ${CUP2DFOLDER}/makefiles/libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISM2DLIBS)

#         3D          #
#######################
CUP3DFOLDER=../_deps/CUP-3D
CUBISM3DCFLAGS=$(shell cat ${CUP3DFOLDER}/makefiles/cup.cflags.txt) -I${CUP3DFOLDER}/source -I${CUP3DFOLDER}/Cubism/include -DCUBISM -Wno-shadow -Wno-float-equal
CUBISM3DLIBS= ${CUP3DFOLDER}/makefiles/libcup.a $(shell cat ${CUP3DFOLDER}/makefiles/cup.libs.txt) -lstdc++fs
swimmerEnvironment3D.o:
	$(KORALICXX) -c _model3D/swimmerEnvironment3D.cpp $(KORALICFLAGS) $(CUBISM3DCFLAGS)
run-vracer-swimmer-3D.o:
	$(KORALICXX) -c -o run-vracer-swimmer-3D.o run-vracer-swimmer.cpp $(KORALICFLAGS) $(CUBISM3DCFLAGS) -DmodelDIM=3
run-vracer-swimmer-3D: run-vracer-swimmer-3D.o swimmerEnvironment3D.o ${CUP3DFOLDER}/makefiles/libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISM3DLIBS)
eval-vracer-swimmer-3D.o:
	$(KORALICXX) -c -o eval-vracer-swimmer-3D.o eval-vracer-swimmer.cpp $(KORALICFLAGS) $(CUBISM3DCFLAGS) -DmodelDIM=3
eval-vracer-swimmer-3D: eval-vracer-swimmer-3D.o swimmerEnvironment3D.o ${CUP3DFOLDER}/makefiles/libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISM3DLIBS)

clean:
	rm -rf run-vracer-swimmer-3D run-vracer-swimmer-2D eval-vracer-swimmer-2D eval-vracer-swimmer-3D *.txt *.o *.ti *.optrpt *.h5 *.dat *.xmf
