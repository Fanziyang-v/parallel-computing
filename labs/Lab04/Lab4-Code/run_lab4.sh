#!/bin/bash

NODE0=10.251.137.194
NODE1=10.251.137.193

# 获得当前目录
CURRENT_DIR=$(cd $(dirname $0); pwd)
EXTRA_LIBS=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs

# 创建dist目录
DIST_DIR=${CURRENT_DIR}/dist

# 构建实验程序
WORK_DIR=${CURRENT_DIR}/hello
cd ${WORK_DIR} && make
ssh ${NODE1} "mkdir -p ${WORK_DIR}" && scp mpi_omp_hello ${NODE1}:${WORK_DIR}

# 进行实验
mpiexec --wd ${WORK_DIR} --host ${NODE0}:2,${NODE1}:2 ./mpi_omp_hello | tee mpi_omp_hello.test0.log

mpiexec --wd ${WORK_DIR} --host ${NODE0},${NODE1} ./mpi_omp_hello | tee mpi_omp_hello.test1.log

mpiexec --wd ${WORK_DIR} --host ${NODE0},${NODE1} -x OMP_NUM_THREADS=3 ./mpi_omp_hello | tee mpi_omp_hello.test2.log

mpiexec --wd ${WORK_DIR} --host ${NODE0},${NODE1} -x OMP_NUM_THREADS=3 ./mpi_omp_hello | tee mpi_omp_hello.test3.log

mpiexec -n 1 --host ${NODE0} --wd ${WORK_DIR} -x OMP_NUM_THREADS=3 ./mpi_omp_hello : -n 2 --host ${NODE1}:2 --wd ${WORK_DIR} -x OMP_NUM_THREADS=2 ./mpi_omp_hello | tee mpi_omp_hello.test3.log

# 进行矩阵乘法实验
WORK_DIR=${CURRENT_DIR}/multigpumatmul
cd ${WORK_DIR} && make
ssh ${NODE1} "mkdir -p ${WORK_DIR}" && scp newfloatmatvec uvafloatmatvec ${NODE1}:${WORK_DIR}
# 进行实验

mpirun --host ${NODE0}:1,${NODE1}:1 --wd ${WORK_DIR} ./newfloatmatvec 400 300 300 400 -p -v | tee newfloatmatvec.test0.log

mpirun --host ${NODE0}:1,${NODE1}:1 --wd ${WORK_DIR} ./uvafloatmatvec 400 300 300 400 -p -v | tee uvafloatmatvec.test2.log


# 进行练习实验
WORK_DIR=${CURRENT_DIR}/practice/mpi_cuda
cd ${WORK_DIR} && make
ssh ${NODE1} "mkdir -p ${WORK_DIR}" && scp vector_addition vector_addition_uva ${NODE1}:${WORK_DIR}

mpirun --host ${NODE0}:1,${NODE1}:1 --wd ${WORK_DIR} -x LD_LIBRARY_PATH=${EXTRA_LIBS}:${LD_LIBRARY_PATH} ./vector_addition | tee vector_addition.test0.log

mpirun --host ${NODE0}:1,${NODE1}:1 --wd ${WORK_DIR} -x LD_LIBRARY_PATH=${EXTRA_LIBS}:${LD_LIBRARY_PATH} ./vector_addition_uva | tee vector_addition_uva.test0.log

WORK_DIR=${CURRENT_DIR}/practice/mpi_openmp
cd ${WORK_DIR} && make
ssh ${NODE1} "mkdir -p ${WORK_DIR}" && scp vector_addition ${NODE1}:${WORK_DIR}

mpiexec --host ${NODE0}:2,${NODE1}:2 --wd ${WORK_DIR} ./vector_addition | tee vector_addition.test0.log

# 最后一个实验
WORK_DIR=${CURRENT_DIR}/SumArray
cd ${WORK_DIR} && make
ssh ${NODE1} "mkdir -p ${WORK_DIR}" && scp mpi_omp_SumArray ${NODE1}:${WORK_DIR}
# 进行实验

mpiexec --host ${NODE0},${NODE1} -x OMP_NUM_THREADS=2 --wd ${WORK_DIR} ./mpi_omp_SumArray | tee mpi_omp_SumArray.test0.log
