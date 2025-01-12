

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        """zip(self.data, args) 将 self.data 和 args 中的元素按索引一一配对，形成一个由元组构成的迭代器，
        每个元组中包含两个元素：累加器中的当前值 a 和对应位置的传入值 b ->[(a1,b1),(a2,b2),...]"""
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]