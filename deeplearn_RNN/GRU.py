import math

from matplotlib import pyplot as plt
from torch import nn
from utils.Accumulator import Accumulator
from utils.Animator import Animator
from utils.RNNModel import RNNModelScratch, RNNModel
from utils.Timer import Timer
from utils.dataPreProcess import load_data_time_machine
import torch
from utils.netEvalue import try_gpu


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size = shape, device=device) * 0.01
    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device=device))
    w_xz, w_hz, b_z = three()
    w_xr, w_hr, b_r = three()
    w_xh, w_hh, b_h = three()
    w_hq = normal((num_hiddens,num_outputs))
    b_q = torch.zeros(num_outputs,device=device)
    params = [w_xz, w_hz, b_z,w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    w_xz, w_hz, b_z , w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q = params
    H, = state
    outputs = []
    for x in inputs:
        Z = torch.sigmoid( (x @ w_xz) + (H @ w_hz) + b_z )
        R = torch.sigmoid( (x @ w_xr) + (H @ w_hr) + b_r )
        H_candidate = torch.tanh( (x @ w_xh) + ((R * H) @ w_hh) + b_h )
        H = Z * H + (1 - Z) * H_candidate
        Y = H @ w_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim = 0), (H,)

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad/batch_size
            param.grad.zero_()

def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, Timer()
    metric = Accumulator(2)  # 训练损失之和,词元数量
    timer.start()
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    timer.stop()
    return math.exp(metric[0] / metric[1]), metric[1] / timer.sum()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    # print(predict('我爱你中国'))
    # print(predict('我爱你中国'))


def main():
    batch_size = 32
    num_steps = 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps,'char')
    vocab_size, num_hiddens, device = vocab.len(), 256, try_gpu()
    num_epochs, lr = 500, 1

    """自己实现---》"""
    # model = RNNModelScratch(vocab_size, num_hiddens, device, get_params, init_gru_state, gru)
    # train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    """《---"""

    """ 简洁实现 ---》"""
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens,num_layers=2)
    model = RNNModel(gru_layer, vocab_size)
    model = model.to(device)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    """《---"""
    plt.show()
if __name__ == '__main__':
    main()