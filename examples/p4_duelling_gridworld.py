'''
Duelling Gridworlds.

Navigate a Grid environment (5x5) which is represented by a pixel screen-output (84x84xRGB) 

The learning happens by two identical conv-NN which update regularly by experience sampling 
'''

import os
import numpy as np

from reinforcement_tools.episode import Transition, Episode, EpisodesBuffer 
from reinforcement_tools.experience import ExperienceSampler

from reinforcement_tools.environments.gridworld import GameEnv
from reinforcement_tools.examples.p4_duelling.actor import Actor    

import logging
logger = logging.getLogger('P4_duelling_gridworld')


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
    
    _print_every = 1 # How often to print status
    _save_every = 5 # How often to save
    _goal = 10    
    
    #Setup the environment
    env = GameEnv(partial=False, size=5)
    
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
    
    logger.info("pretrain finished")
        
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
    
        ep_rewards.append(epi.getTotalReward())
        buffer.add(epi)
        # Drop the probability of a random action :: epsilon greedy
        actor.decay_epsilon()
        
        #Save model now and then
        if n_ep % args.update_freq == 0 and n_ep>0:
            logger.debug('enter training')
            sampler = ExperienceSampler(buffer)
            logger.debug('experiences alligned')
            for num_epoch in range(args.n_epochs_train):
                logger.debug('nn-train epoch: {}'.format(num_epoch))
                s_a_r_s_d = sampler.sample( args.batch_size )
                logger.debug('sampled')
                loss = actor.learn( *s_a_r_s_d )
                losses.append(loss)
                logger.debug('learned')
            # Update the target model with values from the main model
            actor.update_target_graph()
            logger.debug('update nn')
            # Save the model
            if (n_ep+1) % _save_every == 0:
                logger.info('save nn-weights')
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
        