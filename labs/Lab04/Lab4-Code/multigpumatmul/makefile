CC = /usr/bin/gcc
NVCC = /usr/local/cuda/bin/nvcc -ccbin /usr/local/bin/mpicxx
NVCCFLAGS = -Xcompiler -g -w -I.. --std=c++11 -gencode arch=compute_35,code=sm_35
MPI = -I /usr/local/include
LIB = -L /usr/local/lib -L /usr/local/cuda/lib64/stubs -L /usr/local/cuda/lib64 -lmpi
EXTRA_LIBS=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs

BINS:= newfloatmatvec uvafloatmatvec

all: $(BINS)

newfloatmatvec: matvec.cu
	${NVCC} ${NVCCFLAGS} ${MPI} ${LIB} matvec.cu -o newfloatmatvec

uvafloatmatvec: matvec_uva.cu
	${NVCC} ${NVCCFLAGS} ${MPI} ${LIB} matvec_uva.cu -o uvafloatmatvec

test: $(BINS)
	@echo "Test hybrid MPI + Cuda Matrix Multiplication..."
	@mpirun --host csl2wk02:1,csl2wk03:1,csl2wk04:1,csl2wk05:1 -x LD_LIBRARY_PATH=${EXTRA_LIBS}:$LD_LIBRARY_PATH ./newfloatmatvec 400 300 300 400 -p -v

test2: $(BINS)
	@echo "Test hybrid MPI + Cuda Matrix Multiplication with UVA..."
	@mpirun --host csl2wk02:1,csl2wk03:1,csl2wk04:1,csl2wk05:1 -x LD_LIBRARY_PATH=${EXTRA_LIBS}:$LD_LIBRARY_PATH ./uvafloatmatvec 400 300 300 400 -p -v

clean:
	-rm newfloat* uvafloatmatvec


