# C-ImplementationOfBPnetwork
如题，这是一个在C语言中基于梯度下降法实现的反向传播神经网络。
## 简要介绍
麻雀虽小，五脏俱全<br/>
或许唯一不足的就是没有对算法做任何效率上的优化<br/>
要使用神经网络，首先需要打开/code/input.h进行一些设置，
第一栏是神经网络的隐含层数和节点个数，节点个数包括输入输出层。<br/>
第二栏需要设置输入的训练样本数量和测试样本数量，请注意，所有的输入样本必须存放在一个文件里<br/>
第三栏的数据见注释，第四栏的数据是输入输出参数。PRT_RESULT频率控制“每隔多少轮迭代会打印一次对所有测试数据的输出结果”，PRT_ERROR控制每隔多少轮迭代会打印一次均方差。<br/>
之后的几个文件路径含义如下：<br/>
INPUTFILE：存放所有训练样本和测试样本的数据，样本之间不能有标点，长数字内部不能有空格（如123 234会被视作两个数据）<br/>
如果样本有编号，请到/code/input.c中去掉fscanf(...,"*%lf");的注释<br/>
OUTPUTFILE：每隔PRT_RESULT轮迭代会在此打印一遍所有样本的输入输出和累计均方差<br/>
ERRORFILE：每隔PRT_ERROR轮迭代会在此打印一遍迭代轮数和对应的均方差值，用于绘制学习曲线<br/>
TESTFILE:留作未来开发，目前没有用
CHECKFILE：会在这里打印所有样本的输入输出和累计均方差，在这之后，打印“测试样本序号，输出值，误差百分比”，其中测试样本序号是自动从1开始计算的，输出值已经经过反归一化。

## 代码结构
不得不说，这个代码的模块性实在是很差，绝大部分功能都堆在了network.c里，主要是因为这些函数都是直接以BPnetwork这个结构体作为参数的<br/>
以后可能会把神经网络的训练流程函数统一命名之后作为系统部分单独提出来，前向传播、误差信号计算、反向传播和权重更新这四个函数作为计算人部分单独提出来，这里就只保留InitNerwork这个大函数<br/>
1.神经网络结构概览
神经网络本身是一个包含了几乎所有所需参数的大型结构体，但是未来的一些参数可能会用符号常量替代。<br/>

第一部分:运行信息<br/>
这一部分包括了神经网络层数、训练批次大小、学习率、训练迭代次数、训练集大小、input.h中的五个文件对应的指针。
除此之外，还包含reMinMax反归一化结构体，用两个浮点指针成员存储了每个数据对应的最小值和区间。最后就是二重指针input，他指向的每个一重指针都对应一组输入数据（包含训练数据和测试数据）
每组数据都按照先输入后输出的方式依次存储在为这个一重指针分配的空间内。<br/>

第二部分:前向传播信息<br/>
这部分包含了四个浮点二重指针和一个浮点变量，二重指针分别是：节点、偏置、权重和误差。每个二重指针对应的一重指针都对应神经网络的一层，
然后该层的权重矩阵会以行向量从第一行到最后一行顺序存储在该一重指针中，节点向量、偏置向量、误差向量则按照正常方式存储在这个一重指针中。<br/>

第三部分：反向传播信息<br/>
这部分包括三个二重指针，分别是损失信号、偏置微分和权重微分，和第二层按照同样的方式存储每一层的矩阵和向量。

2.神经网络整体流程（main函数）
初始化神经网络、训练神经网络、检测神经网络。






