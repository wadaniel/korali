all: heat2d.o
KORALICFLAGS=`python3 -m korali.cxx --cflags`
ifndef MPICXX
	MPICXX=mpic++
endif

heat2d.o: heat2d.cpp 
	$(MPICXX) -c $< $(KORALICFLAGS) 

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
