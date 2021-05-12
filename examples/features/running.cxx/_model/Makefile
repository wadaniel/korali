all: heat2d.o
KORALICFLAGS=`python3 -m korali.cxx --cflags`

heat2d.o: heat2d.cpp 
	$(CXX) -c $< $(KORALICFLAGS) 

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
