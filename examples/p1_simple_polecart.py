# coding: utf-8

"""
Simple Reinforcement Learning in Tensorflow Part 2-b: 

state is defined as : [position of cart, velocity of cart, angle of pole, rotation rate of pole]
action is in {0,1} >>> left,right
"""


import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym

from reinforcement.common import discount_rewards

class Agent():
    """ A learning Agent for for a generic NN-reinforcement learner, which learns offline from completed episodes
    
    Parameters
    ----------
    state_size : int
        size of the state-tensor (number of state variables/dimensions); determined by the evironment
    action_size : int
        size of the action space (number of viable actions); determined by the enironment
    hidden_size : int >0
        number of neurons in the hidden layer
    learning_rate : float in (0,1)
        Learning rate used in the optimizer (default: 1e-2)
    """
    def __init__(self, lr, s_size, a_size, h_size):
        
        self.a_size = a_size
        
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        
        #define loss and gradients
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        #define the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))
        
        #open a tensorflow-session for reserved operations
        self.tfsession = tf.Session()
        # Launch the tensorflow graph
        self.tfsession.run(tf.global_variables_initializer())
        
    def learn_episodes(self, ep_history_list, gamma):
        """ learn from a list of episodes, themselves holding the transitions during that time
        
        Parameters
        ----------
        ep_history_list : nested list of hierarchy [episodes, steps, (s,a,r,s1)]
        gamma : float
            the discount factor
        """
        
        gradBuffer = self.tfsession.run(tf.trainable_variables())
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
        
        for ep_history in ep_history_list:
            ep_history = np.array(ep_history)
            ep_history[:,2] = discount_rewards(ep_history[:,2], gamma)
            feed_dict={
                    self.reward_holder:ep_history[:,2],
                    self.action_holder:ep_history[:,1],
                    self.state_in:np.vstack(ep_history[:,0])}
            grads = self.tfsession.run(self.gradients, feed_dict=feed_dict)
            for idx,grad in enumerate(grads):
                gradBuffer[idx] += grad
        
        feed_dict = dict(zip(self.gradient_holders, gradBuffer))
        _ = self.tfsession.run(self.update_batch, feed_dict=feed_dict)
        
    def propose_action(self, state):
        """ propose an action given a certain state
        
        Parameters
        ----------
        state : state
        
        Returns
        -------
        action (int) to be performed
        """
        feed_dict = {self.state_in:[state]}
        a_dist = self.tfsession.run( self.output, feed_dict=feed_dict )[0]
        a = np.random.choice(np.arange(self.a_size), p=a_dist)
        return a
    
    
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nepisodes", help="number of episodes to play", type=int, default= 5000)
    parser.add_argument("--maxsteps", help="maximum number of steps in each episode", type=int, default=999)
    parser.add_argument("--updatefreq", help="update the network after evaluation of this many episodes", type=int, default=5)
    parser.add_argument("--gamma", help="gamma for discounting", type=float, default=0.99)
    parser.add_argument("--render", help="enable rendering", type=bool, default=False)
    args = parser.parse_args()  

    #Setup the environment
    env = gym.make('CartPole-v0')
    myAgent = Agent(lr=1e-2, s_size=4, a_size=2, h_size=8) #Load the agent.
    
    ep_reward_list = []
    ep_trans_history_list = []
    
    #--- generate episodes here
    for tot_ep_count in range(args.nepisodes):
        state = env.reset()
        
        ep_reward = 0
        ep_trans_history = []
        
        ep_step_count = 0
        ep_terminated = False
        while ep_step_count < args.maxsteps and not ep_terminated: 
            #get a proposed action from the learner
            action = myAgent.propose_action(state)
            #evaluate the action in the environment
            state_next,reward,ep_terminated,_ = env.step(action)
            trans = [state,action,reward,state_next]
            ep_trans_history.append(trans.copy())
            
            ep_reward += reward
            
            if args.render:
                env.render()
            
            if ep_terminated:
                break
            state = state_next
            ep_step_count += 1
            
        ep_reward_list.append(ep_reward)
        ep_trans_history_list.append(ep_trans_history.copy())
        
        #elavuate if it is time to update
        if tot_ep_count % args.updatefreq == 0 and tot_ep_count != 0:
            myAgent.learn_episodes(ep_trans_history_list[-args.updatefreq:], gamma= args.gamma)
    
        if tot_ep_count % 100 == 0:
            print("Episode {} :: Reward: {} (mean last 100: {})".format(tot_ep_count, ep_reward, np.mean(ep_reward_list[-100:])))
    
