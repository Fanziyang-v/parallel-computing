#!/bin/bash


make clean && make uvafloatmatvec


scp uvafloatmatvec 10.251.137.193:/home/lenovo/openmpi/Lab04/Lab4-Code/multigpumatmul

mpirun --host 10.251.137.194:1,10.251.137.193:1 --wd /home/lenovo/openmpi/Lab04/Lab4-Code/multigpumatmul ./uvafloatmatvec 400 300 300 400 -p -v