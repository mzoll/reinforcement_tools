'''
Created on Aug 6, 2018

@author: marcel.zoll
'''

import numpy as np
import random

class Transition():
    """ A transition is the state, the action taken, the immediate reward, and the new state"""
    def __init__(self, s, a, r, s1):
        self.s = s
        self.a = a
        self.r = r
        self.s1 = s1

class Episode():
    """ An episode is a Series of transitions of arbitrary length or until the Episode is terminated """
    def __init__(self, start_state):
        self.length = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_state = start_state
        self.done = False
    def getTotalReward(self):
        return np.sum(self.rewards)
    def addTransition(self, trans, done=False):
        """ Add a next transition; signal the terminating 'done' state of the Episode """
        if self.done:
            raise Exception("This episode has already concluded, no more Transitions can be added")
        self.states.append(trans.s)
        self.actions.append(trans.a)
        self.rewards.append(trans.r)
        self.next_state = trans.s1
        self.length += 1
        self.done= done
    def getTransition(self, i):
        """ get a specific transition (zero-centered counting) """
        if self.length <= i:
            raise ValueError("the Transition at that index does not exist")
        if self.length < i-1: 
            t = Transition( self.states[i], self.actions[i], self.rewards[i], self.states[i+1]) 
        else:
            t = Transition( self.states[i], self.actions[i], self.rewards[i], self.next_state) 
        return t
    def getArrays(self):
        """ get alligned arrays of all transitions: state, action, reward, next_state, done"""
        done = np.zeros(self.length)
        if self.done:
            done[self.length-1] = 1
        return np.array(self.states), np.array(self.actions), np.array(self.rewards), np.array( self.states[1:] + [self.next_state] ), np.array( done )
    def sample(self):
        return self.getTransition( random.randint(0, self.length-1) )

class EpisodesBuffer:
    """ hold  a number of episodes in a rolling buffer """
    def __init__(self, buffer_size=1000):
        """ Data structure used to hold game experiences """
        self.buffer_size = buffer_size
        self.buffer = []
    def add(self, episode):
        """ Adds list of experiences to the buffer """
        self.buffer.append(episode)
        self.buffer = self.buffer[-self.buffer_size:]
    def sample(self, size):
        """ Returns a sample of experiences from the buffer """
        sample = random.choice(self.buffer, size, replace=False)
        return sample
    def getArrays(self):
        """ get alligned arrays of all transitions: state, action, reward, next_state, done"""
        s_a_r_n_d = [], [], [], [], []
        for epi in self.buffer:
            _s_a_r_n_d = epi.getArrays()
            for i in range(5):
                s_a_r_n_d[i].append(_s_a_r_n_d[i])
        return np.vstack(s_a_r_n_d[0]), np.hstack(s_a_r_n_d[1]), np.hstack(s_a_r_n_d[2]), np.vstack(s_a_r_n_d[3]), np.hstack(s_a_r_n_d[4])
