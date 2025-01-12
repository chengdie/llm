from matplotlib import pyplot as plt


def xyAxis_draw(x, y , xlabel = 'x', ylabel = 'y', title = 'draw with x,y data'):
    # 绘制图表
    plt.figure(figsize=(6, 3))  # 设置图表大小
    plt.plot(x, y, label='Data')  # 绘制数据
    plt.xlabel(xlabel)  # 设置x轴标签
    plt.ylabel(ylabel)  # 设置y轴标签
    plt.title(title)  # 设置图表标题
    plt.legend()  # 显示图例
    plt.grid(True)
    plt.show()  # 显示图表

def xyAxis_draw_multi( y , linestyles, markers, labels = None, xlabel = 'x', ylabel = 'y', title = 'draw with x,y data'):
    for i in range(len(y)):
        plt.plot(y[i], label = labels[i] if labels is not None else f'{i}', linestyle=linestyles[i], marker=markers[i])
    plt.xlabel(xlabel)  # 设置x轴标签
    plt.ylabel(ylabel)  # 设置y轴标签
    plt.title(title)  # 设置图表标题
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()  # 显示图例
    plt.show()  # 显示图表

#  n可以为1，亦可以为k
def n_step_ahead_prediction(time_n, x_n, xlabel='time', ylabel='x', title='Time Series and Predictions', legend=None, xlim=[1, 1000], figsize=(6, 3)):
    # 将 PyTorch 张量转换为 NumPy 数组
    for i in range(len(time_n)):
        time_n[i] = time_n[i].detach().numpy()
        x_n[i] = x_n[i].detach().numpy()
    # 绘制图表
    plt.figure(figsize=figsize)  # 设置图表大小
    for i in range(len(time_n)):
        if legend:
            plt.plot(time_n[i], x_n[i], label=legend[i])
        else:
            plt.plot(time_n[i], x_n[i])
    plt.xlabel(xlabel)  # 设置x轴标签
    plt.ylabel(ylabel)  # 设置y轴标签
    plt.title(title)  # 设置图表标题
    if legend:
        plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.xlim(xlim)  # 设置x轴范围
    plt.show()  # 显示图表