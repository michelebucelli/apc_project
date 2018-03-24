CXX = g++
CXXFLAGS = -Wall -std=c++14

OBJECTS = main.o kmeans_base.o kmeans.o kmeans_sdg.o

EXE = main.out

all : $(EXE)

$(EXE) : $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run : $(EXE)
	./$(EXE)

%.o : kmeans_base.h kmeans.h kmeans_sdg.h

clean :
	rm -f *.o

distclean : clean
	rm -f $(EXE)
