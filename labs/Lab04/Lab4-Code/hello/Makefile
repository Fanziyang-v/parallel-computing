BINS:= mpi_omp_hello

all: $(BINS)

mpi_omp_hello: mpi_omp_hello.c
	mpic++ -fopenmp -o mpi_omp_hello mpi_omp_hello.c

test: $(BINS)
	@echo "Test hybrid MPI + OpenMP Hello..."
	@mpiexec  --host csl2wk02:2,csl2wk03:2 ./mpi_omp_hello

test1: $(BINS)
	@echo "Test hybrid MPI + OpenMP Hello..."
	@mpiexec --host csl2wk02,csl2wk03 ./mpi_omp_hello

test2: $(BINS)
	@echo "Test hybrid MPI + OpenMP Hello..."
	@mpiexec --host csl2wk02,csl2wk03 -x OMP_NUM_THREADS=3 ./mpi_omp_hello

test3: $(BINS)
	@echo "Test hybrid MPI + OpenMP Hello..."
	@mpiexec -n 1 --host csl2wk02 -x OMP_NUM_THREADS=3 ./mpi_omp_hello : -n 2 --host csl2wk03:2 -x OMP_NUM_THREADS=2 ./mpi_omp_hello

clean:
	@rm -f $(BINS)