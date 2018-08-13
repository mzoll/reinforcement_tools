'''
Created on Aug 1, 2018

@author: marcel.zoll
'''

import datetime as dt
from realtimemachine.classes import Result

from reinforcement.model import ReforceModel
from reinforcement.polecart.actor import PoleCartActor

from realtimemachine.prime_building.bypass import BypassPrimeBuilder

class PoleCartModel(ReforceModel):
    def __init__(self):
        ReforceModel.__init__(self, 'PoleCartVanilla', 42)
        self.actor = PoleCartActor()
        self.primeBuilders = [ BypassPrimeBuilder('PCPrimeBuilder', self.actor._state_vars) ]
        self.stateBuilders = self.primeBuilders[0].getStateBuilders() 
        
    def score(self, prime):
        action_values = self.actor.actions_prob(prime.data)
        r = Result(prime.uid, prime.targetid, dt.datetime.now(), prime.routingkey, prime.meta)
        r.results = action_values
        return  r
        
    def prepare(self):
        self.actor.prepare()
        return self
        
    def getPrimeBuilders(self):    
        return self.primeBuilders
    def getStateBuilders(self):
        return self.stateBuilders