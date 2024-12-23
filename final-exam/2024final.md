# 2024 年 HITSZ 并行计算期末考试题目回忆版

**一、选择题（每题 2 分，共20分）**

（1）什么是并行效率？

（2）大规模图形渲染，用什么计算模型比较合适？（ **SIMD**/MIMD/SISD/MISD ）

（3）根据 Flynn 分类法，GPU是什么模型？（ **SIMD**/MIMD/SISD/MISD）

（4）通过 OpenMP 的什么指令可以使得一段代码块一次只有一个线程执行？（**critical**/atomic/parallel/barrier）

（5）两个长度为2000的向量，利用 CUDA 进行向量加法操作，线程块的大小为 512，总线程数是多少？（1024/2024/**2048**/4096）

（6）忘了。

（7）强扩展性和弱扩展性的概念。

（8）一个 CUDA 程序执行时，共有 100 个线程块，每个线程块的线程数为 512，Kernel 创建一个局部变量，则共有多少个不同的变量？（51200）

（9）上一题的背景，若创建的是共享存储变量，则共有（100）个不同的变量。

（10）调用MPI_Send(prt_a, 1000, MPI_FLOAT, 2000, 4, MPI_COMM_WORLD) 传递了 4000 个字节的数据，则 传输的单位数据元素长度是多少字节？（1B/2B/**4B**/8B）



**二、填空题（每空1分，共15分）**

（1）4个处理器，并行效率不少于 90%，串行程序执行时间 Ts=72秒，则并行程序执行时间至多不超过（   ）秒？

（2）模型分类，填 分布式/共享内存模型

TCP/UDP：分布式内存

Posix（p threads）：共享内存

OpenMP：共享内存

MPI：分布式内存

CUDA：共享内存

（3）MPI_Send 与（  ）和（   ）形成消息对完成消息传递

（4）

给一段程序

| T0                                                         | T1                                                         | T2                                                         | T3                                                         |
| ---------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| a=4;b=16                                                   | a=4;b=16                                                   | a=4;b=16                                                   | a=4;b=16                                                   |
| MPI_Reduce(&a, &b, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) | MPI_Reduce(&a, &b, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) | MPI_Reduce(&a, &b, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) | MPI_Reduce(&a, &b, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) |
| MPI_Reduce(&b, &a, 1, MPI_INT, MPI_SUM, 1, MPI_COMM_WORLD) | MPI_Reduce(&b, &a, 1, MPI_INT, MPI_SUM, 1, MPI_COMM_WORLD) | MPI_Reduce(&b, &a, 1, MPI_INT, MPI_SUM, 1, MPI_COMM_WORLD) | MPI_Reduce(&b, &a, 1, MPI_INT, MPI_SUM, 1, MPI_COMM_WORLD) |

程序执行完毕后，进程 0 的 a= （  ），b=（  ），进程 1 的 a=（  ），b=（  ）

（5）每个线程块共有 512 个线程，共有 16个线程块，则每个线程块共有（  ）个 warp，总共有 （  ）个 warp。



**三、简答题（每题 5 分，共 20 分）**

（1）4个处理器，要使得程序的加速比不小于 16，分别应用 Amdahl 定律和 Gustafson 定律，求最大的串行比例。

（2）绑定任务和非绑定任务的区别是什么？列举出发生任务暂停和任务调度点的情况。

（3）分析以下两种情况，GPU 对分支指令的执行方式，(i) 1 个 warp 的分支不同，(ii) 1 个 warp 的分支相同，2 个warp 的分支不同。

（4）简要介绍共享内存的 Bank Conflict。



**四、分析题（每题 15分，共 45 分）**：

（1）给一段 OpenMP 程序，分别写出循环展开并行化代码和任务并行化代码，并分析两种方式的负载均衡。

```c
void f(int i)
{
    int start = i * (i + 1) / 2, finish = start + i;
    float return_val = 0.0;
    
    for (int j = start; j <= finish; j++)
        return_val += cos(j)
    return return_val;
}

float sum = 0.0;
for (i = 0; i <= n; i++)
    sum += f(i)
```

（2）编写 CUDA Kernel 函数，实现两个 nxn 矩阵的 Elementwise 乘法运算，实现两个 Kernel 函数，第一个 Kernel 函数中，一个线程计算一个输出元素，第二个 Kernel 函数中，一个线程计算一列的输出元素。并且给出两种方式的调用参数设置，最后比较两种 Kernel 函数的优劣。

（3）分析树形 AllReduce 和环形 AllReduce 通信算法的时间复杂度，并比较两种方式的优劣。假设共有 P 个计算节点，共有 M 个待归约单精度浮点数，通信延迟为 α，单位字节的传输时间为 β，m 字节数据的通信时间由以下公式给出：t(m) = α + βm。

