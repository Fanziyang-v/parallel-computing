BINS:= mpi_omp_SumArray

all: $(BINS)

mpi_omp_SumArray: mpi_omp_SumArray.c
	mpic++ -std=c99 -fopenmp -o mpi_omp_SumArray mpi_omp_SumArray.c

test: $(BINS)
	@echo "Test hybrid MPI + OpenMP Array Sum..."
	@mpiexec --host csl2wk02,csl2wk03 -x OMP_NUM_THREADS=2 ./mpi_omp_SumArray

clean:
	@rm -f $(BINS)