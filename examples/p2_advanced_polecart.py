import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt


class Actor(object):
    def __init__(self, discount=0.97):
        self.discount = discount
        self.policy_grad = self.policy_gradient()
        self.val_grad = self.value_gradient()
    def prepare(self):
        self.tfsession = tf.Session()
        self.tfsession.run(tf.global_variables_initializer())
    def policy_gradient(self):
        with tf.variable_scope("policy"):
            state = tf.placeholder("float",[None,4], name='state')
            
            params = tf.get_variable("policy_parameters",[4,2])
            linear = tf.matmul(state,params)
            probabilities = tf.nn.softmax(linear, name='probabilities')
    
            actions = tf.placeholder("float",[None,2], name='actions')
            advantages = tf.placeholder("float",[None,1], name='advantages')
            good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions),reduction_indices=[1])
            eligibility = tf.log(good_probabilities) * advantages
            loss = -tf.reduce_sum(eligibility)
            optimizer = tf.train.AdamOptimizer(0.01).minimize(loss, name='optimizer')
            return probabilities, state, actions, advantages, optimizer
    
    def value_gradient(self):
        with tf.variable_scope("value"):
            state = tf.placeholder("float",[None,4], name='state')
            
            w1 = tf.get_variable("w1",[4,10])
            b1 = tf.get_variable("b1",[10])
            h1 = tf.nn.relu(tf.add(tf.matmul(state,w1), b1))
            w2 = tf.get_variable("w2",[10,1])
            b2 = tf.get_variable("b2",[1])
            calculated = tf.add(tf.matmul(h1,w2), b2, name='calculated')
            
            newvals = tf.placeholder("float",[None,1], name='newvals')
            diffs = calculated - newvals
            loss = tf.nn.l2_loss(diffs)
            optimizer = tf.train.AdamOptimizer(0.1).minimize(loss, name='optimizer')
            return calculated, state, newvals, optimizer, loss
        
    def action_probs(self, obs_vector):
        probs = self.tfsession.run('policy/probabilities:0', feed_dict={'policy/state:0': obs_vector})
        return probs
    def learn_episode(self, transitions):
        #do training steps
        _discount = self.discount
        
        states = []
        actions_oh = []
        advantages = []
        update_vals = []
        
        for index, trans in enumerate(transitions):
            obs, action, reward = trans
    
            # calculate discounted monte-carlo return
            future_reward = 0
            future_transitions = len(transitions) - index
            decrease = 1
            for index2 in range(future_transitions):
                future_reward += transitions[(index2) + index][2] * decrease
                decrease = decrease * _discount
            obs_vector = np.expand_dims(obs, axis=0)
            currentval = self.tfsession.run('value/calculated:0', feed_dict={'value/state:0': obs_vector})[0][0]
    
            # advantage: how much better was this action than normal
            advantages.append(future_reward - currentval)
    
            # update the value function towards new return
            update_vals.append(future_reward)
            
            #stack the observations
            states.append(obs)
            #stack the actions as one hots            
            actionblank = np.zeros(2)
            actionblank[action] = 1
            actions_oh.append(actionblank)
            
        # update value function
        update_vals_vector = np.expand_dims(update_vals, axis=1)
        self.tfsession.run('value/optimizer', feed_dict={'value/state:0': states, 'value/newvals:0': update_vals_vector})
        # update the policy
        advantages_vector = np.expand_dims(advantages, axis=1)
        self.tfsession.run('policy/optimizer', feed_dict={'policy/state:0': states, 'policy/advantages:0': advantages_vector, 'policy/actions:0': actions_oh})


def run_episode(env, actor):
    
    observation = env.reset()
    totalreward = 0
    transitions = []

    for _ in range(200):
        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = actor.action_probs(obs_vector)
        action = 0 if random.uniform(0,1) < probs[0][0] else 1 #int(binary) action
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break
    #do training steps
    actor.learn_episode(transitions)
    return totalreward


#=============================================================================
env = gym.make('CartPole-v0')
#env.monitor.start('cartpole-hill/', force=True)

actor = Actor()
actor.prepare()

#dry-run
for i in range(2000):
    reward = run_episode(env, actor)
    if reward == 200:
        print ("reward 200")
        print(i)
        break

#live  
t = 0
for _ in range(1000):
    reward = run_episode(env, actor)
    t += reward
print (t / 1000)
