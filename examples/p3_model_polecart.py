
# coding: utf-8

# # Simple Reinforcement Learning with Tensorflow: Part 3 - Model-Based RL
# In this iPython notebook we implement a policy and model network which work in tandem to solve the CartPole reinforcement learning problem. To learn more, read here: https://medium.com/p/9a6fe0cce99
# 
# For more reinforcment learning tutorials, see:
# https://github.com/awjuliani/DeepRL-Agents

# ### Loading libraries and starting CartPole environment

# In[3]:


import numpy as np

import gym

# model initialization
state_dim = 4 # state dimensionality
action_dim = 1 # action dimensionality
action_space = [0,1] # ^= {left, right}
action_size = len(action_space)
done_dim = 1 # state dimensionality
done_space = [0,1] # ^= {not done, done}
reward_dim= 1

# hyperparameters
_modellearn_bs = 3 # Batch size when learning from model
_envlearn_bs = 3 # Batch size when learning from real environment



def discount(r, gamma = 0.99):
    r_disc = []
    _r = 0
    for r in reversed(r):
        _r = _r *gamma + r
        r_disc.append(_r)
    return np.array(list(reversed(r_disc)))

def standardize(vec):
    return (np.array(vec).astype('float32') - np.mean(vec)) / np.std(vec)

def comp_advantages(actions, rewards, policies):
    discrewards = standardize(discount(ep_rewards))
                
    advantages = []
    for policy, action, disc_reward in zip(policies, actions, discrewards):
        policy[action] = disc_reward
        advantages.append(policy)
    return advantages


def env_rand_initstate():
    return np.random.uniform(-0.05, 0.05, [4])


#=============================
# define TRnasitions and Episodes
#=============================
class Transition():
    def __init__(self, s, a, r, s1):
        self.s = s
        self.a = a
        self.r = r
        self.s1 = s1

class Episode():
    def __init__(self, start_state):
        self.length = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_state = start_state
        self.done = False
    def addTransition(self, trans, done=False):
        if self.done:
            raise Exception("This episode has already concluded, no more Transitions can be added")
        self.states.append(trans.s)
        self.actions.append(trans.a)
        self.rewards.append(trans.r)
        self.next_state = trans.s1
        self.length += 1
        self.done= done
    def getTotalReward(self):
        return np.sum(self.rewards)
    def getTransition(self, i):
        if self.length <= i:
            raise ValueError("the Transition at that index does not exist")
        if self.length < i-1: 
            t = Transition( self.states[i], self.actions[i], self.rewards[i], self.states[i+1]) 
        else:
            t = Transition( self.states[i], self.actions[i], self.rewards[i], self.next_state) 
        return t
    def getArrays(self):
        done = np.zeros(self.length)
        if self.done:
            done[self.length-1] = 1
        return np.vstack(self.states), np.vstack(self.actions), np.vstack(self.rewards), np.vstack( self.states[1:] + [self.next_state] ), np.vstack( done )


#===============================
# define Policy Episodes
#===============================
class Policy():
    def __init__(self, a_dist):
        self.a_dist = a_dist

class PolicyEpisode():
    def __init__(self):
        self.policies = []
    def addPolicy(self, policy):
        self.policies.append(policy.a_dist)
    @property
    def length(self):
        return len(self.policies)


import keras

#from keras import backend as K
#K.set_image_dim_ordering('tf')

#print(keras.__version__)

#from keras.layers.advanced_activations import Softmax
from keras.layers import Input, Dense, Concatenate, Softmax
from keras.models import Model as KModel
from keras.optimizers import Adam

from sklearn.preprocessing import OneHotEncoder

#========================================
# define the actor
#========================================
class Actor(object):
    def __init__(self):
        self.model = self._compileModel()
    def _compileModel(self, hidden_size=256, learning_rate= 1e-3):
        #sequencetail_block; in: series of integers, zero padded in front
        state_input = Input(shape=(state_dim,), dtype='float32', name='state_input')
        _ = Dense(hidden_size, activation= 'relu')(state_input)
        action_output = Dense(2, activation='softmax', name='action_output')(_)
        # = Softmax()(_)
        
        kmodel = KModel(inputs=[state_input],
                      outputs=[action_output])
    
        kmodel.compile(optimizer=Adam(lr=learning_rate), loss='mse')
        return kmodel
    
    def propose_action(self, state):
        """ returns an int for a probolistic (payout is prob) chosen action """
        a_dist = self.action_rewards(state)
        a = np.random.choice(np.arange(self.action_size), p=a_dist)
        return a
    def propose_best_action(self, state):
        """ return an int for the best action to take (highest payout) """
        a_dist = self.action_rewards(state)
        a = np.argmax(a_dist)
        return a
    def action_rewards(self, state):
        """ returns rewards per action as a vector"""
        action_biprob = self.model.predict( state.reshape(1,4) )
        return action_biprob.reshape(len(action_space))
    
    def learn(self, states, advantages):
        states = np.array(states)
        advantages = np.array(advantages)
        self.model.train_on_batch(states, advantages)


#========================================
# define the actor
#========================================

class EnvModel():
    def __init__(self):
        self.model = self._compileModel()
        
        self.enc = OneHotEncoder(2, sparse=False)
        self.enc.feature_indices_ = np.array([0, 2])
        self.enc.n_values_ = 2
    def _compileModel(self, hidden_size=256, learning_rate= 1e-3):
        #sequencetail_block; in: series of integers, zero padded in front
        state_input = Input(shape=(state_dim,), dtype='float32', name='state_input')
        action_input = Input(shape=(len(action_space),), dtype='float32', name='action_input')
        input_layer = Concatenate()([state_input, action_input])
        
        _ = Dense(hidden_size)(input_layer)
        _ = Dense(hidden_size)(_)
        
        reward_output = Dense(reward_dim, activation='relu', name='reward_output')(_)
        nextstate_output = Dense(state_dim, activation='relu', name='nextstate_output')(_)
        done_output = Dense(done_dim, activation='sigmoid', name='done_output')(_)
        
        kmodel = KModel(inputs=[state_input, action_input],
                      outputs=[reward_output, nextstate_output, done_output])

        kmodel.compile(optimizer=Adam(lr=learning_rate), loss='mse')
        return kmodel
    
    def step(self, state0, action0):
        """ make one transition-step in the modelled environment
        (state0[array], action0[scalar]) -> (step1[array], reward1[scalar], done1[scalar])
        """
        action_oh = self.enc.transform( [[action]] )

        feed_dict = {'state_input': np.array(state0).reshape(1, state_dim), 'action_input': action_oh }
        reward_arr, step1_arr, done_arr = self.model.predict(feed_dict)
        
        step1 = step1_arr[0]
        reward = reward_arr[0]
        done = done_arr[0]
        
        #pebble stuff into the correct limits
        step1[0] = np.clip(step1[0], -2.4, 2.4) #DANGER outside knowledge Here
        step1[2] = np.clip(step1[2], -0.4, 0.4)
        done = np.clip(done[0], 0., 1.)
        reward = reward[0] 
        return step1, reward, done
    
    def learn(self, states, actions, rewards, nextstates, dones):
        """ learn from a completed or partially completed episode to model the enivironment
        
        Parameters
        ----------
        states : np.array shape(n, state_dim)
        actions : np.array shape(n)
        rewards : np.array shape(n)
        next_states : np.array shape(n, state_dim)
        dones : np.array shape(n)
        """
        actions_oh = self.enc.transform( actions )
        
        in_dict= {'state_input': states, 'action_input': actions_oh}
        out_dict = {'nextstate_output': nextstates, 'reward_output': rewards, 'done_output': dones}        
        loss = self.model.train_on_batch( in_dict, out_dict )
        return loss

    
#==========================
# main block
#=========================

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nepisodes", help="number of episodes to play", type=int, default= 500)
    parser.add_argument("--maxsteps", help="maximum number of steps in each episode", type=int, default=300)
    parser.add_argument("--updatefreq", help="update the network after evaluation of this many episodes", type=int, default=5)
    parser.add_argument("--gamma", help="gamma for discounting", type=float, default=0.99)
    parser.add_argument("--decay_rate", help="decay rate for past episodes", type=float, default=0.99)
    parser.add_argument("--render", help="enable rendering", type=bool, default=False)
    args = parser.parse_args()  

    #Setup the environment
    env = gym.make('CartPole-v0')

    
    actor = Actor()
    envModel = EnvModel()

    real_episodes = 0
    episodes = []    
    pepisodes = []
        
    batch_size = _envlearn_bs
    batch_accu = 0
    nep_switch = 10000
    drawFromModel = False # When set to True, will use model for observations
    trainTheModel = True # Whether to train the model
    trainThePolicy = False # Whether to train the policy


    episode_number = 0
    while episode_number <= args.nepisodes:
        episode_number += 1
        
        observation = env.reset()
        ep = Episode(observation)
        pep = PolicyEpisode()
        
        ep_step_count = 0
        ep_terminated = False
        ep_reward = 0
        while ep_step_count < args.maxsteps and not ep_terminated:
            
            if args.render: 
                env.render()
            
            action_rew = actor.action_rewards(observation)
            #action_rew /= np.sum(action_rew)
            action = np.random.choice(action_space, p=action_rew)
            #print('Action', action)
            
            # step the  model or real environment and get new measurements
            if drawFromModel:
                next_observation, reward, doneP = envModel.step(observation, action)
                done = doneP > 0.1 or ep_step_count >= args.maxsteps #coax the done state into a better regime
            else:
                next_observation, reward, done, _ = env.step(action)
            
            ep_terminated = done
            ep.addTransition( Transition( observation, action, reward, next_observation ), done )
            pep.addPolicy( Policy( action_rew ))
            
            ep_step_count += 1
            
            observation = next_observation
            
        #episode terminated
        print("Model" if drawFromModel else "Env", "Episode {} :: length : {} total reward : {}".format(episode_number, ep.length, ep.getTotalReward()))
        episodes.append(ep)
        pepisodes.append(pep)
        
        if not drawFromModel: 
            real_episodes += 1

        ep_states, ep_actions, ep_rewards, ep_nextstates, ep_done = ep.getArrays()

        if trainTheModel:
            envModelLoss = envModel.learn( ep_states, ep_actions, ep_rewards, ep_nextstates, ep_done )
            print(envModelLoss)
            
        if episode_number > nep_switch:
            batch_accu += 1
            if batch_accu >= batch_size:
                if trainThePolicy:

                    acc_states = []
                    acc_advantages = []
                    
                    for i in range(len(episodes)-batch_size,len(episodes)):
                        epi = episodes[i]
                        pepi = pepisodes[i]
                        
                        epi_states, epi_actions, epi_rewards, epi_nextstates, ep_done = epi.getArrays()
                        pepi_policies = pepi.policies
                        
                        advantages = comp_advantages(epi_actions, epi_rewards, pepi_policies)
                    
                        acc_states.append(epi_states)
                        acc_advantages.append(advantages)
                    
                    acc_states = np.vstack(acc_states)
                    acc_advantages = np.vstack(acc_advantages)
                     
                    actor.learn(acc_states, acc_advantages)
                
                drawFromModel = not drawFromModel
                trainTheModel = not trainTheModel
                trainThePolicy = not trainThePolicy
                batch_accu = 0

