'''
Created on Apr 3, 2018

@author: marcel.zoll
'''

import unittest

import numpy as np
from reinforcement_tools.environments.gridworld import GameEnv

#import matplotlib.pyplot as plt

class Test(unittest.TestCase):

    def testGameEnv(self):
        env= GameEnv(partial=False, size=5)
        
        state= env.reset()
        
        for i in range(500):
            a = np.random.randint(env.actions)
            state = env.step(a)
            #plt.imshow(state)
            #plt.draw()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()