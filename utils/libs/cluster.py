import numpy as np
from tqdm import tqdm

class KMeans:
    def __init__(self, center_numbers=2):
        self.center_numbers = center_numbers
        self.centers = None

    def fit(self, data):

