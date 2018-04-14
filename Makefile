CXX = mpicxx
CXXFLAGS = -Wall -std=c++14

OBJECTS = main.o kmeans_base.o kmeans.o kmeans_sgd.o kmeans_seq.o
OUTPUT = output.txt
EXE = main.out

NP = 2

METHOD = kmeans
TEST = g500000
K = 6
ARGS = -t $(TEST) -k $(K) -m $(METHOD) --purity

all : $(EXE)

$(EXE) : $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run : $(EXE)
	mpiexec -np $(NP) ./$(EXE) $(ARGS) >$(OUTPUT)

plot : $(OUTPUT)
	@ gnuplot -dc plotScript.plt -p 2>/dev/null

%.o : kmeans_base.h kmeans.h kmeans_sgd.h kmeansseq.h

clean :
	rm -f *.o

distclean : clean
	rm -f $(EXE)
	rm -f output.txt
