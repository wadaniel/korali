all: heat2d.o

CXXFLAGS += -march=native -D NOALIAS -O3 -g

heat2d.o: heat2d.cpp 
	$(CXX) -c $< $(CXXFLAGS) 

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
