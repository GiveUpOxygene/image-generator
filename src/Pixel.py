import random
import scipy as sp

class Pixel:
    def __init__(self, x, y, tab):
        """
        Initialize a Pixel object with coordinates and pixel values.

        :param x: X-coordinate of the pixel.
        :param y: Y-coordinate of the pixel.
        :param tab: 3D numpy array of pixel values.
        """
        self.x = x
        self.y = y
        self.tab = tab
        self.mu = None
        self.sigma = None
        self._fill_law_params()
        

    def _fill_law_params(self):
        """
        Fill the parameters for the statistical law.
        """
        shap = Pixel.shapiro_wilk_test(self.tab)
        if (shap[1] < 0.05):
            # normal distribution
            self.mu, self.sigma = sp.stats.norm.fit(self.tab)
            self.sigma = 0.7 * self.sigma
        elif(shap[1] >= 0.05):
            # uniform distribution
            self.mu = min(self.tab)
            self.sigma = max(self.tab) - self.mu
        else:
            raise ValueError("Unexpected result from Shapiro-Wilk test.")
        
    def __str__(self):
        """
        String representation of the Pixel object.
        """
        return f"Pixel({self.x}, {self.y}, mu={self.mu}, sigma={self.sigma})"
    
    def generate_pixel_value(self):
        """
        Generate a pixel value based on the statistical law defined by the parameters.
        """
        if self.mu is not None and self.sigma is not None:
            return self.normal_distribution()
        elif self.a is not None and self.b is not None:
            return self.uniform_distribution()
        else:
            raise ValueError("Statistical parameters are not set correctly.")



    def normal_distribution(self):
        a, b = (0 - float(self.mu)) / self.sigma, (255 - float(self.mu)) / self.sigma
        return int(round(sp.stats.truncnorm.rvs(a, b, loc=self.mu, scale=self.sigma)))

    def uniform_distribution(self):
        return random.uniform(min(self.tab), max(self.tab))

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