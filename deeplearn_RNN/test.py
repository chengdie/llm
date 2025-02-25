import torch
import matplotlib.pyplot as plt


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """显示矩阵热图"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

# 示例数据
matrices = torch.randn(2, 3, 4, 4)  # 2 行 3 列的 4x4 矩阵
xlabel = 'Keys'
ylabel = 'Queries'
titles = ['Matrix 1', 'Matrix 2', 'Matrix 3']

# 显示热图
show_heatmaps(matrices, xlabel, ylabel, titles, figsize=(6, 4), cmap='Blues')
plt.show()