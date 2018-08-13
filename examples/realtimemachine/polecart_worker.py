""" Example how to build an agent that ballances a pole on a cart """

import os, sys, pathlib
  
import numpy as np
import datetime as dt
import pickle
import logging
import time

#--- pipeline and model assembly
from realtimemachine.model.collection import ModelCollection

#--- realtimemachine components
from realtimemachine.machine.rtworker import RealTimeWorker

#--- request generation
from realtimemachine.classes import Request
    
import common_tools.units

from reinforcement.polecart.model import PoleCartModel
import uuid
import gym

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrainepisodes", help="number of episodes to train, before deploying the agent", type=int, default= 500)
    parser.add_argument("--nepisodes", help="number of requests to publish", type=int, default= 100)
    parser.add_argument("--maxsteps", help="maximum number of steps in each episode", type=int, default= 999)
    args = parser.parse_args()    

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
    
    _routingId_0 = 1
    
    mc = ModelCollection()
    mc.rk_model_dict[_routingId_0] = pcm
    
    #=================================
    input(">>> Models prepared; Setup RTWorker ...")
    #=================================
    
    #=== REALTIMEMACHINE
    rtw = RealTimeWorker(
            modelcollection = mc,
            statestorehandler = None,
            resultstorehandler = None)
    rtw.setup()
    
    #=================================
    input(">>> RTWorker set up; start processing requests...")
    #=================================

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
            request = Request(request_id, tid, dt.datetime.now(), str(_routingId_0), meta= {'nepisode': i},
                    payload = dict(zip(pcm.actor._state_vars, pcstate)) )
            
            if ep_step_count==0:
                request.payload['Reset'] = True
            if prev_ep_terminated:
                request.payload['NewSession'] = True
                prev_ep_terminated = False
            
            request.timestamp = dt.datetime.now() #for turn-around timining
            response = rtw.process(request)
            response.timestamp = dt.datetime.now() #for turn-around timining
            
            request_id += 1 #raise the request counter
            
            requests.append(request)
            responses.append(response)
            
            #--- extract and perform action in enironment
            action = np.argmax( [ response.payload[k] for k in pcm.actor._action_space ] )
            #action = response.payload['action']

            pcstate_next,reward,ep_terminated,_ = env.step(action)
            
            ep_step_count += 1 #raise the step counter
            trans = [pcstate,action,reward,pcstate_next]
            ep_trans_history.append(trans.copy())
            
            if ep_terminated:
                prev_ep_terminated = True
                
                break
            pcstate = pcstate_next
    
    print("Nepisodes {}, latest nsteps {}".format(i, ep_step_count))
    print("Processed {} requests".format(len(responses)))
    
    print('METRIC: Received Responses: ', len(responses))
    dt_turnaround = [ (res.timestamp -req.timestamp).total_seconds()  for req,res in zip(requests, responses) ]
    #there is a startup lag, so throw away the first n responses/requests
    print('METRIC: turn-around [ms]:\n avg={:.3}, min={:.3}, lower.1={:.3}, median={:.3}, upper.1={:.3}, max={:.3},'.format(np.average(dt_turnaround)*1000.,
                                                                      np.min(dt_turnaround)*1000.,
                                                                      np.percentile(dt_turnaround, 10)*1000,
                                                                      np.median(dt_turnaround)*1000.,
                                                                      np.percentile(dt_turnaround, 90)*1000,
                                                                      np.max(dt_turnaround)*1000., ) )
    input(">>> Teardown RTWorker ...")
    rtw.teardown()

    input(">>> RTMachine stopped; exit ...")
    