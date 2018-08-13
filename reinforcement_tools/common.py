'''
Created on Aug 1, 2018

@author: marcel.zoll
'''

import numpy as np

def discount_rewards(r, gamma = 0.99):
    """ take 1D float array of rewards and compute discounted reward
    
    Parameters
    ----------
    r : array of float
        the rewards
        
    Returns
    -------
    discounted rewards
     """
    r_disc = []
    _r = 0
    for r in reversed(r):
        _r = _r *gamma + r
        r_disc.append(_r)
    return np.array(list(reversed(r_disc)))

