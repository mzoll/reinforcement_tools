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
        for _ in range(500):
            a = np.random.randint(env.actions)
            next_state, reward, done = env.step(a)
            #plt.imshow(state)
            #plt.draw()
            state = next_state

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()