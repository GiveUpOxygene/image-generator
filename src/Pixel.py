import random
import scipy as sp

class Pixel:
    def __init__(self, law: str):
        self.law = law

    @staticmethod
    def normal_distribution(mean, std_dev):
        return random.gauss(mean, std_dev)

    @staticmethod
    def uniform_distribution(start, end):
        return random.uniform(start, end)

    @staticmethod
    def random_integer(start, end):
        return random.randint(start, end)
    
    @staticmethod
    def shapiro_wilk_test(data: list):
        """
        Perform the Shapiro-Wilk test for normality.
        Returns a tuple of (W statistic, p-value).
        """
        return sp.stats.shapiro(data)
    
    @staticmethod
    def ks_test(data1: list, data2: list):
        """
        Perform the Kolmogorov-Smirnov test for two samples.
        Returns a tuple of (D statistic, p-value).
        """
        return sp.stats.ks_2samp(data1, data2)
    
    @staticmethod
    def chi_square_test(observed: list, expected: list):
        """
        Perform the Chi-Square test for goodness of fit.
        Returns a tuple of (Chi-Square statistic, p-value).
        """
        return sp.stats.chisquare(observed, expected)
    
    @staticmethod
    def exp_test(data: list):
        """
        Perform the Exponential test.
        Returns a tuple of (lambda estimate, p-value).
        """
        return sp.stats.expon.fit(data)
    

    # utiliser ks_test pour tout?
    # créer des fonctions représentant chaque loi?