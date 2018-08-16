'''
Created on Aug 13, 2018

@author: marcel.zoll
'''

import os
import numpy as np
from keras.models import Model as KModel
from keras.layers import Conv2D, Dense, Flatten, Input, Lambda
import keras.backend as K


import logging
logger = logging.getLogger('GridworldActor')


class Actor(object):
    """ An actor for the Gridworld game field
    
    Parameters
    ----------
    img_dims : tuple (x,y,z)
        the dimensions of the observable (screen) input
    action_size : int > 0
        number of actions that can be performed
    final_layer_size : int > 0
        size of the final convolutional layer
    gamma : float in (0..1)
        discount factor for rewards
    tau : float in (0..1)
        the internal learning rate for target NN from the main NN
    epsilon_target : (default: 0.1)
        the initial and deterministic probability to choose a random action instead of a the highest rewarding one (epsilon greedy)
    epsilon_decay : float >0 (default: 0.005)
        the decay rate of epsilon, evetually diminishing it to epsilon_target, applied by each call to `decay_epsilon()`
    """
    def __init__(self, img_dims, action_size, final_layer_size, gamma=0.97, tau=0.1, epsilon_target = 0.1, epsilon_decay=0.005):
        self._img_dims = img_dims
        self._action_size = action_size
        self._final_layer_size = final_layer_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon_target = epsilon_target
        self.epsilon_decay = epsilon_decay
        
        self.epsilon = epsilon_target
        self.main_qn = self.compile_model(self._img_dims, self._action_size, self._final_layer_size)
        self.target_qn = self.compile_model(self._img_dims, self._action_size, self._final_layer_size)
        
    def decay_epsilon(self):
        """ decay the random chance for exploration (episoln greedy) """
        self.epsilon -= (self.epsilon - self.epsilon_target) * self.epsilon_decay
        
    def compile_model(self, img_dims, action_size, final_layer_size):
        """
        Parameters
        ----------
        img_dims : tuple form (dim_x, dim_y, dim_rgb)
        final_layer_size
        """
        # The input image of the game is 84 x 84 x 3 (RGB) 
        state_input = Input(shape=img_dims, name="states_input")

        # There will be four layers of convolutions performed on the image input
        # A convolution take a portion of an input and matrix multiplies
        # a filter on the portion to get a new input (see below)
        _ = Conv2D(
            filters=32,
            kernel_size=[8,8],
            strides=[4,4],
            activation="relu",
            padding="valid",
            name="conv1")(state_input)
        _ = Conv2D(
            filters=64,
            kernel_size=[4,4],
            strides=[2,2],
            activation="relu",
            padding="valid",
            name="conv2")(_)
        _ = Conv2D(
            filters=64,
            kernel_size=[3,3],
            strides=[1,1],
            activation="relu",
            padding="valid",
            name="conv3")(_)

        stream_AC = Conv2D(
            filters= int(final_layer_size / 2),
            kernel_size=[7,7],
            strides=[1,1],
            activation="relu",
            padding="valid",
            name="conv4_ac")(_)
        
        stream_VC = Conv2D(
            filters= int(final_layer_size / 2),
            kernel_size=[7,7],
            strides=[1,1],
            activation="relu",
            padding="valid",
            name="conv4_vc")(_)
        
        stream_AC = Flatten(name="advantage_flatten")(stream_AC)
        advantage = Dense(self._action_size, name="advantage_final")(stream_AC)
        adv_delta = Lambda(lambda adv: adv - K.mean(adv, axis=1, keepdims=True), name="advantage_delta")(advantage)
        
        stream_VC = Flatten(name="value_flatten")(stream_VC)
        value = Dense(1,name="value_final")(stream_VC)

        action_output = Lambda(lambda val_advd: val_advd[0] + val_advd[1],name="action_out")([value, adv_delta])
        
        model = KModel(state_input, action_output)
        model.compile("adam","mse")
        model.optimizer.lr = 0.0001
        return model
    def action_probs(self, state):
        action_probs = self.main_qn.predict(np.array([state]))
        return action_probs
    def propose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self._action_size)
        else:
            action = np.argmax(self.action_probs(state))
        return action
    def learn(self, train_state, train_action, train_reward, train_next_state, train_done):
        # Our predictions (actions to take) from the main Q network
        target_q = self.target_qn.predict(train_state) #&TRAIN
        
        # The Q values from our target network from the next state
        target_q_next_state = self.main_qn.predict(train_next_state) #&Main
        train_next_state_action = np.argmax(target_q_next_state, axis=1)
        #train_next_state_action = train_next_state_action.astype(np.int)

        # Q value of the next state based on action
        train_next_state_values = target_q_next_state[:, train_next_state_action]

        # Tells us whether game over or not
        # We will multiply our rewards by this value
        # to ensure we don't train on the last move
        train_gamecont = ~train_done.astype(bool)

        # Reward from the action chosen in the train batch
        actual_reward = train_reward + (train_gamecont * self.gamma * train_next_state_values )
        target_q[:, train_action] = actual_reward
        
        # Train the main model
        loss = self.main_qn.train_on_batch(train_state, target_q)
        return loss
    def update_target_graph(self):
        updated_weights = (np.array(self.main_qn.get_weights()) * self.tau) + \
            (np.array(self.target_qn.get_weights()) * (1. - self.tau))
        self.target_qn.set_weights(updated_weights)
    def _save_weights(self, main_weights_file, target_weights_file):
        self.main_qn.save_weights(main_weights_file)
        self.target_qn.save_weights(target_weights_file)
    def _load_weights(self, main_weights_file, target_weights_file):
        self.main_qn.load_weights(main_weights_file)
        self.target_qn.load_weights(target_weights_file)
        