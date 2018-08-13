'''
Created on Aug 13, 2018

@author: marcel.zoll
'''

import numpy as np
from .episode import EpisodesBuffer

class ExperienceSampler():
    def __init__(self, ep_buffer):
        self.arrs = ep_buffer.getArrays()
    def sample(self, size):
        """ sample by slicing """
        idx = np.random.choice(len(self.arrs[0]), size=size, replace=False)
        return [ arr[idx,] for arr in self.arrs ]
