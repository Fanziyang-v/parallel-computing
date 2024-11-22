CC = /usr/bin/gcc
NVCC = /usr/local/cuda/bin/nvcc -ccbin /usr/local/bin/mpicxx
NVCCFLAGS = -Xcompiler -g -w -I.. --std=c++11 -gencode arch=compute_35,code=sm_35
MPI = -I /usr/local/include
LIB = -L /usr/local/lib -L /usr/local/cuda/lib64/stubs -L /usr/local/cuda/lib64 -lmpi
EXTRA_LIBS=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs



vector_addition: vector_addition.cu
	${NVCC} ${NVCCFLAGS} ${MPI} ${LIB} vector_addition.cu -o vector_addition

vector_addition_uva: vector_addition_uva.cu
	${NVCC} ${NVCCFLAGS} ${MPI} ${LIB} vector_addition_uva.cu -o vector_addition_uva

test: vector_addition
	@echo "Test MPI+CUDA vector_addition..."
	@mpirun --host csl2wk02:1,csl2wk03:1 -x LD_LIBRARY_PATH=${EXTRA_LIBS}:$LD_LIBRARY_PATH ./vector_addition

test2: vector_addition_uva
	@echo "Test MPI+CUDA vector_addition_uva..."
	@mpirun --host csl2wk02:1,csl2wk03:1 -x LD_LIBRARY_PATH=${EXTRA_LIBS}:$LD_LIBRARY_PATH ./vector_addition_uva

clean:
	-rm vector_addition vector_addition_uva


