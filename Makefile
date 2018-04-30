CXX = mpicxx
OPTIMIZE = F

ifeq ($(OPTIMIZE),T)
CXXFLAGS += -Wall -std=c++14 -O3 -DNDEBUG
else
CXXFLAGS += -Wall -std=c++14 -DNDEBUG
endif

OBJECTS = point.o kmeans_base.o kmeans_parallel.o kmeans.o kmeans_sgd.o kmeans_seq.o main.o
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
	nice -20 mpiexec -np $(NP) ./$(EXE) $(ARGS) >$(OUTPUT)

plot : $(OUTPUT)
	@ octave plotScript.m

%.o : point.h kmeans_base.h kmeans_parallel.h kmeans.h kmeans_sgd.h kmeansseq.h

clean :
	rm -f *.o

distclean : clean
	rm -f $(EXE)
	rm -f output.txt
