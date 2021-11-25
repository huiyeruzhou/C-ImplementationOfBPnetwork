# C-ImplementationOfBPnetwork
如题，这是一个在C语言中基于梯度下降法实现的反向传播神经网络。
## 文件内容说明
input.txt-存放所有输入数据，每行依次包含所有的输入值和输出值，按浮点类型存储<br/>
output.txt-存放所有输出数据，每组输出第一行显示数据的迭代次数和序号，第二行显示输入值，第三行显示神经网络输出结果<br/>
error.csv-存放损失函数，第一行为表头“error”“round”，之后每行存储一个批次内的累计均方差和该批次对应的训练次数（每个样本计算一次）<br/>
——这个文件可以用excel打开，用于快速绘制学习曲线图<br/>
### /code
input.h-存放了神经网络运行的关键数据<br/>
network.c-神经网络的主要实现<br/>
calculate.c-矩阵乘法等计算的实现<br/>
### /test
test.c-向input.txt写入输入数据。<br/>


