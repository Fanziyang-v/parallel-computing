BINS:= vector_addition

all: $(BINS)

vector_addition: vector_addition.c
	mpic++ -fopenmp -o vector_addition vector_addition.c

test: $(BINS)
	@echo "Test MPI+OpenMP vector_addition..."
	@mpiexec  --host csl2wk02:2,csl2wk03:2 ./vector_addition

clean:
	@rm -f $(BINS)