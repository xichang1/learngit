import torch
import torch.nn as nn #定义torch.nn为nn
import torch.nn.functional as F #引用torch.nn.functional并将其定义F
N, D_in, H, D_out = 64, 1000, 100, 10  #训练64个数据，输入层单元个数为1000个，隐藏层单元个数为100个，输出层单元个数为10个
learning_rate = 0.001 
x = torch.rand(N,D_in)
y = torch.rand(N,D_out)
class TwoLayerNet(torch.nn.Module):    #建立一个模型
    def __init__(self, D_in, H, D_out): #初始化
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    #建立一个Liner类，bias表示偏置项，建立Wx+b
    def forward (self, x):    #forward 是torch.nn.Module定义好的模块，表示前向传
     h_relu = self.linear1(F.relu(x))
   
     y_pred = self.linear2(F.relu(h_relu))
     return y_pred

net = TwoLayerNet (D_in, H, D_out)     #实例化类
criterion =torch.nn.MSELoss(reduction='mean')   #损失函数,均方损失函数(xi-yi)2
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)   #数据优化
for t in range(200):    #计算两百次
    optimizer.zero_grad() #梯度归零
    #通过x传送到模型中计算出预测的y
    y_pred = net(x)
    #计算和打印损失
    loss = criterion(y_pred,y) 
    loss.backward()  #反向传播
    optimizer.step()  #更新参数
    print(t, loss.item())  #这里打印的是loss的平均值

