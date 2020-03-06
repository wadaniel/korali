all: model.o load_data.o
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`

# %.o: %.cpp %.hpp load_data.hpp

load_data.o: load_data.cpp  load_data.hpp
	@echo "Compiling: $< -> $@"
	$(KORALICXX) $(CXXFLAGS) $(INCLUDES) $(KORALICFLAGS) -MP -MMD -c $< -o $@

#load_data.o: load_data.cpp  load_data.hpp
	
#	$(KORALICXX) -c $< $(KORALICFLAGS)


model.o: model.cpp model.hpp 
	@echo "Compiling: $< -> $@"
	$(KORALICXX) $(CXXFLAGS) $(INCLUDES) $(KORALICFLAGS) -MP -MMD -c $< -o $@

#	$(KORALICXX) -c $< $(KORALICFLAGS)

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
