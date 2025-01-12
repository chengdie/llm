import torch

def evaluate_loss(net, data_iter, loss):
    net.eval()  # 将模型设置为评估模式
    sum_loss,total_samples = 0.0, 0
    with torch.no_grad():  # 在这个上下文中，不计算梯度
        for X, y in data_iter:
            y_hat = net(X)  # 获取模型的预测值
            l = loss(y_hat, y)  # 计算损失值
            sum_loss += l.sum().item()  # 累加损失值
            total_samples += y.shape[0]  # 累加样本数量

    net.train()  # 将模型设置回训练模式
    return sum_loss / total_samples  # 返回平均损失值

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')