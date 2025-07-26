import random

class Pixel:
    def __init__(self, value=0):
        self.value = value

    @staticmethod
    def normal_distribution(mean, std_dev):
        return random.gauss(mean, std_dev)

    @staticmethod
    def uniform_distribution(start, end):
        return random.uniform(start, end)

    @staticmethod
    def random_integer(start, end):
        return random.randint(start, end)