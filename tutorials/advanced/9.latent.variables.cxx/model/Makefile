all: model.o load_data.o some_samplers.o test_model
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

# %.o: %.cpp %.hpp load_data.hpp

test_model: test_model.o model.o load_data.o
	$(KORALICXX) -o $@ $^ $(KORALILIBS)

test_model.o: test_model.cpp test_model.hpp model.hpp load_data.hpp
	@echo "Compiling: $< -> $@"
	$(KORALICXX) $(CXXFLAGS) $(INCLUDES) $(KORALICFLAGS) -MP -MMD -c $< -o $@

load_data.o: load_data.cpp
	@echo "Compiling: $< -> $@"
	$(KORALICXX) $(CXXFLAGS) $(INCLUDES) $(KORALICFLAGS) -MP -MMD -c $< -o $@


%.o: %.cpp %.hpp
	$(KORALICXX) -c $(KORALICFLAGS) $<


#load_data.o: load_data.cpp  load_data.hpp
	
#	$(KORALICXX) -c $< $(KORALICFLAGS)


model.o: model.cpp model.hpp load_data.hpp
	@echo "Compiling: $< -> $@"
	$(KORALICXX) $(CXXFLAGS) $(INCLUDES) $(KORALICFLAGS) -MP -MMD -c $< -o $@

#	$(KORALICXX) -c $< $(KORALICFLAGS)

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
