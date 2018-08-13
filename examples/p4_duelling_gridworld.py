from __future__ import division
import numpy as np
from keras.models import Model as KModel
from keras.layers import Conv2D, Dense, Flatten, Input, Lambda
import keras.backend as K

import matplotlib.pyplot as plt
import scipy.misc
import os

import gridworld
from reinforcement.episode import Transition, Episode, EpisodesBuffer 

class Actor(object):
    def __init__(self, img_dims, action_size, final_layer_size, gamma, tau, epsilon_target = 0.1, epsilon_decay=0.005):
        self._img_dims = img_dims
        self._action_size = action_size
        self._final_layer_size = final_layer_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon_target = epsilon_target
        self.epsilon_decay = epsilon_decay
        
        self.epsilon = 1.
        self.main_qn = self.compile_model(self._img_dims, self._action_size, self._final_layer_size)
        self.target_qn = self.compile_model(self._img_dims, self._action_size, self._final_layer_size)
        
    def decay_epsilon(self):
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
        advantage = Dense(env.actions,name="advantage_final")(stream_AC)
        adv_delta = Lambda(lambda adv: adv - K.mean(adv, axis=1, keepdims=True), name="adv_delta")(advantage)
        
        stream_VC = Flatten(name="value_flatten")(stream_VC)
        value = Dense(1,name="value_final")(stream_VC)

        action_output = Lambda(lambda val_advd: val_advd[0] + val_advd[1],name="action_out")([value, adv_delta])
        
        model = KModel(state_input, action_output)
        model.compile("adam","mse")
        model.optimizer.lr = 0.0001
        return model
    def update_target_graph(self):
        updated_weights = (np.array(self.main_qn.get_weights()) * self.tau) + \
            (np.array(self.target_qn.get_weights()) * (1. - self.tau))
        self.target_qn.set_weights(updated_weights)
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
        train_next_state_action = train_next_state_action.astype(np.int)
        
        # Tells us whether game over or not
        # We will multiply our rewards by this value
        # to ensure we don't train on the last move
        train_gamecont = train_done == 0

        # Q value of the next state based on action
        train_next_state_values = target_q_next_state[:, train_next_state_action]

        # Reward from the action chosen in the train batch
        actual_reward = train_reward + (self.gamma * train_next_state_values * train_gamecont)
        target_q[:, train_action] = actual_reward
        
        # Train the main model
        loss = self.main_qn.train_on_batch(train_state, target_q)
        return loss
    def _save_weights(self, main_weights_file, target_weights_file):
        self.main_qn.save_weights(main_weights_file)
        self.target_qn.save_weights(target_weights_file)
    def _load_weights(self, main_weights_file, target_weights_file):
        self.main_qn.load_weights(main_weights_file)
        self.target_qn.load_weights(target_weights_file)
        
    

class ExperienceSampler():
    def __init__(self, ep_buffer):
        self.arrs = ep_buffer.getArrays()
    def sample(self, size):
        idx = np.random.choice(len(self.arrs[0]), size=size, replace=False)
        states, actions, rewards, next_states, dones = [ arr[idx,] for arr in self.arrs]
        return states, actions, rewards, next_states, dones
    


#============================================================= MAIN ==========================
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--nepisodes", help="number of episodes to play", type=int, default= 500)
    #parser.add_argument("--maxsteps", help="maximum number of steps in each episode", type=int, default=300)
    args = parser.parse_args()  

    args.n_ep_pretrain = 100
    args.n_ep_live = 1000
    args.max_epi_steps = 50
    
    args.update_freq = 5
    args.n_epochs_train = 20
    
    args.gamma = 0.99 #gamma
    args.tau = 1.
    args.epsilon_target = 0.1
    args.epsilon_decay = 0.005
    
    args.batch_size = 64
    
    args.load_model = False
    
    _path = "./models" # Path to save our model to
    _main_weights_file = _path + "/main_weights.h5" # File to save our main weights to
    _target_weights_file = _path + "/target_weights.h5" # File to save our target weights to
    
    _print_every = 50 # How often to print status
    _save_every = 5 # How often to save
    _goal = 10    
    
    #Setup the environment
    env = gridworld.GameEnv(partial=False, size=5)
    
    actor = Actor(
            img_dims=(84,84,3),
            action_size = env.actions,
            final_layer_size= 512,
            gamma = args.gamma, 
            tau= args.tau,
            epsilon_target= args.epsilon_target,
            epsilon_decay= args.epsilon_decay)
    
        
    # Setup path for saving
    if not os.path.exists(_path):
        os.makedirs(_path)
    
    #--- load Model weights
    if args.load_model:
        actor._load_weights(_main_weights_file, _target_weights_file)
    
    #---setup buffer
    buffer = EpisodesBuffer(1000)
    losses = []
    ep_rewards = []
    
    
    #--- pretrain phase ----------------------------------------------------
    for n_ep in range(args.n_ep_pretrain):
        
        state = env.reset()
        epi = Episode(state)
        
        epi_step = 0
        epi_terminated = False
        
        while not epi_terminated and epi_step < args.max_epi_steps:
            epi_step += 1
            
            action = actor.propose_action(state) #np.random.randint(env.actions)
            next_state, reward, done = env.step(action)
            
            epi.addTransition( Transition(state, action, reward, next_state), done )
            
            state = next_state
            epi_terminated = done
    
        ep_rewards.append(epi.getTotalReward())
        buffer.add(epi)
    
    print("pretrain finished")
        
    #--- go live with model --------------------------------------------------------
    actor.epsilon = 0.6
    for n_ep in range(args.n_ep_live):
        state = env.reset()
        epi = Episode(state)
        
        epi_step = 0
        epi_terminated = False
        
        while not epi_terminated and epi_step < args.max_epi_steps:
            epi_step += 1
            
            action = actor.propose_action(state)
            next_state, reward, done = env.step(action)
            
            epi.addTransition( Transition(state, action, reward, next_state) )
            
            state = next_state
            epi_terminated = done
    
        buffer.add(epi)
        # Drop the probability of a random action :: epsilon greedy
        actor.decay_epsilon()
        
        #Save model now and then
        if n_ep % args.update_freq == 0 and n_ep>0:
            sampler = ExperienceSampler(buffer)
            for num_epoch in range(args.n_epochs_train):
                loss = actor.learn( *sampler.sample( args.batch_size ) )
                losses.append(loss)
            # Update the target model with values from the main model
            actor.update_target_graph()
                    
            # Save the model
            if (n_ep+1) % _save_every == 0:
                actor._save_weights(_main_weights_file, _target_weights_file)
    
        # Print progress
        if n_ep % _print_every == 0:
            mean_loss = np.mean(losses[-(_print_every*args.n_epochs_train):])
            mean_reward = np.mean(ep_rewards[-_print_every:])
            print("Num episode: {} Mean reward: {:0.4f} Prob random: {:0.4f}, Loss: {:0.04f}".format(
                n_ep, mean_reward, actor.epsilon, mean_loss))
            if mean_reward >= _goal:
                print("Training complete!")
                break


    
        
        
        