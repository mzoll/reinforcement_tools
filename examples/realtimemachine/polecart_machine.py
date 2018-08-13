"""
Demonstrate that a realtimemache can work.

We are using dummy components, so that actual work is done, but there is no real merit behind 
some of the tasks as the spew out random numbers rather than to compute a viable score

Perform the following steps:

    * Load all needed Imports
    * Specify two distinct dummy models
    * Fit the dummy models
    * Prepare some hobo-requests
    * Set Up a RealTimeMachine with Agents and Workers
    * Start the components
    * Process the Request, Read the Responses
    * Stop the components (print out some statistics)
    * Tear down the RealTimeMachine (cleanup)
    
Remeber that local thrid party components need to be up and running:
    
    * Redis
    * RabbitMq
"""

import os, sys, pathlib
  
import numpy as np
import datetime as dt
import pickle
import logging
import time
import copy

from common_tools.helpers.credentials import credkey

import gym
from reinforcement.polecart.model import PoleCartModel

#--- realtimemachine components
from realtimemachine.machine.rtagent import RealTimeAgent
from realtimemachine.machine.rtpool import RealTimePool
from realtimemachine.machine.rtmachine import RealTimeMachine

from realtimemachine.model.collection import ModelCollection

#--- cache
from common_tools.localcache.redis import RedisCache_Master
from common_tools.localcache.memcached import MemcachedCache_Master
from common_tools.localcache.managed import ManagedCache_Master

from common_tools.units import units

#--- request generation
from realtimemachine.classes import Request
import uuid
    
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--nworkers", help="number of workers (consumer-threads) to pool", type= int, default= 2)
    parser.add_argument("--ntrainepisodes", help="number of episodes to train, before deploying the agent", type=int, default= 100)
    parser.add_argument("--nepisodes", help="number of requests to publish", type=int, default= 200)
    parser.add_argument("--maxsteps", help="maximum number of steps in each episode", type=int, default= 200)
    parser.add_argument("--cachemode", help="which cache to use: {0: managed_cache, 1: memcache, 2: redis}", type=int, default=1)
    parser.add_argument("--render", help="enable rendering", type=bool, default=True)
    args = parser.parse_args()    
    
    #determine which cache to use
    if args.cachemode == 0:
        cache = ManagedCache_Master()
    elif args.cachemode == 1:
        cache = MemcachedCache_Master(val_ttl=12, wait_on_insert=False)
    elif args.cachemode == 2:
        cache = RedisCache_Master(val_ttl=12, wait_on_insert=False)
    else:
        raise ValueError("Unknown cachemode {}".format(args.cachemode))
    
    #Connection properties for pika: we need to tell where to connect to the server
    import pika
    PIKA_CONPARAMS = pika.ConnectionParameters(
        host= credkey('RABBITMQ', 'HOST', 'localhost'),
        port= int(credkey('RABBITMQ', 'PORT', '5672')),
        virtual_host= '/',
        credentials = pika.PlainCredentials(
            credkey('RABBITMQ', 'USERNAME', 'guest'),
            credkey('RABBITMQ', 'PASSWORD', 'guest')))
    
     #==========================
    input(">>> Setup a PoleCartAgent")
    #==========================
    
    pcm = PoleCartModel()
    pcm.prepare()
    
    #==================================
    input(">>> Train the Model ...")
    #==================================
    
    env = gym.make('CartPole-v0') #, dtype=np.float32)
    
    class train_args():
        nepisodes = args.ntrainepisodes
        maxsteps = args.maxsteps
        updatefreq = 5
        gamma = 0.99
    
    ep_reward_list = []
    ep_trans_history_list = []
    
    #--- generate episodes here
    for tot_ep_count in range(train_args.nepisodes):
        state = env.reset()
        
        ep_reward = 0
        ep_trans_history = []
        
        ep_step_count = 0
        ep_terminated = False
        while ep_step_count < train_args.maxsteps and not ep_terminated: 
            #get a proposed action from the learner
            action = pcm.actor.propose_action(state)
            #evaluate the action in the environment
            state_next,reward,ep_terminated,_ = env.step(action)
            trans = [state,action,reward,state_next]
            ep_trans_history.append(trans.copy())
            
            ep_reward += reward
            
            if ep_terminated:
                break
            state = state_next
            ep_step_count += 1
            
        ep_reward_list.append(ep_reward)
        ep_trans_history_list.append(ep_trans_history.copy())
        
        #elavuate if it is time to update
        if tot_ep_count % train_args.updatefreq == 0 and tot_ep_count != 0:
            pcm.actor.learn_episodes(ep_trans_history_list[-train_args.updatefreq:], gamma= train_args.gamma)
    
    print("trained {} episodes; latest episode length {}".format(tot_ep_count+1, ep_step_count))
    
        #==================================
    input(">>> Pack up Models ...")
    #==================================
    
    routingId_0 = 1
    
    mc = ModelCollection()
    mc.rk_model_dict[routingId_0] = pcm
    

    input(">>> Setup RTMachine ...")
    
    #=== REALTIMEMACHINE
    #generate and setup a RealTimeMachine(Host)
    rtm = RealTimeMachine('RTMachine', PIKA_CONPARAMS)
    rtm.setup()
    
    #generate and setup a RealTimeAgent
    response_received = []
    def on_response_callback(response):
        """ simply append each response to the list """
        response_received.append(response)
        
    rta = RealTimeAgent('IMP', PIKA_CONPARAMS, [str(routingId_0)], on_response_callback)
    rta.setup()
    rta.bind(rtm.request_exchange_name, rtm.response_exchange_name)
        
    #generate and setup a RealTimePool
    rtp = RealTimePool(
            'RTPool',
            PIKA_CONPARAMS,
            [str(routingId_0)],
            modelcollection = mc,
            statestorehandler = cache,
            resultstorehandler = None,
            nworkers= args.nworkers)
    rtp.setup()
    rtp.bind(rtm.request_exchange_name, rtm.response_exchange_name)

    input(">>> RTMachine set up; start the components ...") 
    rtp.start()
    rta.start()
    
    input(">>> Start to juggle")
    
    requests = []
    responses = []
    
    request_id = 0
    tid = uuid.uuid4().hex
    
    for i in range(args.nepisodes):    
        pcstate = env.reset()
        ep_step_count = 0
        ep_terminated = False
        prev_ep_terminated = True
        while ep_step_count < args.maxsteps and not ep_terminated:
            #--- prepare the request querying an action
            request = Request(request_id, tid, dt.datetime.now(), str(routingId_0), meta= {'nepisode': i},
                    payload = dict(zip(pcm.actor._state_vars, pcstate)) )
            
            #mix in the right signals
            if ep_step_count==0:
                request.payload['Reset'] = True
            if prev_ep_terminated:
                request.payload['NewSession'] = True
                prev_ep_terminated = False
            
            request.timestamp = dt.datetime.now() #for turn-around timining
            
            rta.publish(request)
            
            while not len(response_received):
                time.sleep(0.01)
            
            response = response_received.pop()
            request_id += 1 #raise the request counter
            
            requests.append(request)
            responses.append(response)
            
            #--- extract and perform action in enironment
            action = np.argmax( [ response.payload[k] for k in pcm.actor._action_space ] )

            pcstate_next,reward,ep_terminated,_ = env.step(action)
            
            ep_step_count += 1 #raise the step counter
            
            if args.render:
                env.render()
            
            if ep_terminated:
                prev_ep_terminated = True
                break
            pcstate = pcstate_next
    
    input(">>> Requests published; Stop Agent, Stop Pool ...")
    rta.stop()
    rtp.stop()
    
    print('METRIC: Received Responses: ', len(responses))
    dt_turnaround = [ (res.timestamp -req.timestamp).total_seconds()  for req,res in zip(requests, responses) ]
    #there is a startup lag, so throw away the first n responses/requests
    print('METRIC: turn-around [ms]:\n avg={:.3}, min={:.3}, lower.1={:.3}, median={:.3}, upper.1={:.3}, max={:.3},'.format(np.average(dt_turnaround)*1000.,
                                                                      np.min(dt_turnaround)*1000.,
                                                                      np.percentile(dt_turnaround, 10)*1000,
                                                                      np.median(dt_turnaround)*1000.,
                                                                      np.percentile(dt_turnaround, 90)*1000,
                                                                      np.max(dt_turnaround)*1000., ) )
    input(">>> Teardown Pool, Agent, Machine ...")
    rtp.teardown()
    rta.teardown()
    rtm.teardown()

    input(">>> RTMachine stopped; exit ...")
    