import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.total_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.total_time += time.time() - self.start_time

    def sum(self):
        return self.total_time