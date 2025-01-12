import torch
from torch import nn
from utils.funcdraw import *
from utils.dataPreProcess import *
from utils.netEvalue import *


# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

def train(net, train_iter, optimizer, loss, epochs, lr):
    for epoch in range(epochs):
        for x, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(x), y)
            l.sum().backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {evaluate_loss(net, train_iter, loss):f}')

def main():
    T = 1000
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    # xyAxis_draw(time, x) #数据展示
    tau = 4
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i:T - tau + i]
    labels = x[tau:].reshape((-1, 1))
    """ 开始训练 """
    batch_size, n_train = 16, 600
    train_iter = load_array(features, labels, n_train, batch_size, True)
    net = get_net()
    epochs, lr =  5, 0.01
    # 平方损失。注意：MSELoss计算平方误差时不带系数1/2
    loss = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(net.parameters(), lr)
    train(net, train_iter, optimizer, loss, epochs, lr)

    #单步预测
    # n_step_ahead_prediction([time,time[tau:]], [x, net(features)], xlim=[1, 1000],figsize=(6, 3))
    #k步预测
    # multistep_preds = torch.zeros(T)
    # multistep_preds[: n_train + tau] = x[: n_train + tau]
    # for i in range(n_train + tau, T):
    #     multistep_preds[i] = net(
    #         multistep_preds[i - tau:i].reshape((1, -1)))
    # n_step_ahead_prediction([time, time[tau:], time[n_train + tau:]], [x, net(features),  multistep_preds[n_train + tau:]], 'time',
    #                         'x', legend=['data', '1-step preds', 'multistep preds'],xlim=[1, 1000], figsize=(6, 3))

    # 1-4-16-64步预测对比
    max_steps = 64
    features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
    # 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
    for i in range(tau):
        features[:, i] = x[i: i + T - tau - max_steps + 1]
    # 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau:i]).reshape(-1)
    steps = (1, 4, 16, 64)
    n_step_ahead_prediction([time[tau + i - 1: T - max_steps + i] for i in steps],
                            [features[:, (tau + i - 1)] for i in steps], 'time',
                            'x', legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000], figsize=(6, 3))
if __name__ == '__main__':
    main()






