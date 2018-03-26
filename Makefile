CXX = mpicxx
CXXFLAGS = -Wall -std=c++14

OBJECTS = main.o kmeans_base.o kmeans.o kmeans_sgd.o
OUTPUT = output.txt
EXE = main.out

TEST = s1.txt

all : $(EXE)

$(EXE) : $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run : $(EXE)
	./$(EXE) >$(OUTPUT)

plot : $(OUTPUT)
	gnuplot -dc plotScript.plt -p

%.o : kmeans_base.h kmeans.h kmeans_sgd.h

clean :
	rm -f *.o

distclean : clean
	rm -f $(EXE)
	rm -f output.txt
