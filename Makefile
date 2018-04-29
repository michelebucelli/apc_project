CXX = mpicxx
CXXFLAGS += -Wall -std=c++14 -O3 -DNDEBUG

OBJECTS = main.o kmeans_base.o kmeans.o kmeans_sgd.o kmeans_seq.o
OUTPUT = output.txt
EXE = main.out

NP = 2

METHOD = kmeans
TEST = g1M-20-5
K = 5
ARGS = -t $(TEST) -k $(K) -m $(METHOD) --purity --no-output

all : $(EXE)

$(EXE) : $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run : $(EXE)
	mpiexec -np $(NP) ./$(EXE) $(ARGS) >$(OUTPUT)

plot : $(OUTPUT)
	@ octave plotScript.m

%.o : kmeans_base.h kmeans.h kmeans_sgd.h kmeansseq.h

clean :
	rm -f *.o

distclean : clean
	rm -f $(EXE)
	rm -f output.txt
