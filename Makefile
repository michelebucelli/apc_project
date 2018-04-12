CXX = mpicxx
CXXFLAGS = -Wall -std=c++14

OBJECTS = main.o kmeans_base.o kmeans.o kmeans_sgd.o
OUTPUT = output.txt
EXE = main.out

NP = 4
ARGS = -t s1 -k 15 --purity -m kmeans

all : $(EXE)

$(EXE) : $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run : $(EXE)
	mpiexec -np $(NP) ./$(EXE) $(ARGS) >$(OUTPUT)

plot : $(OUTPUT)
	gnuplot -dc plotScript.plt -p

%.o : kmeans_base.h kmeans.h kmeans_sgd.h

clean :
	rm -f *.o

distclean : clean
	rm -f $(EXE)
	rm -f output.txt
