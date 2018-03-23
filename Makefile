CXX = g++
CXXFLAGS = -Wall -std=c++14

OBJECTS = main.o kmeans_base.o kmeans.o kmeans_sdg.o

EXE = main

all : $(EXE)

$(EXE) : $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run : $(EXE)
	./$(EXE)

%.o : kmeans_base.h kmeans.h kmeans_sdg.h

clean :
	rm -f *.o

disclean : clean
	rm -f $(EXE)
