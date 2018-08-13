'''
Created on Aug 1, 2018

@author: marcel.zoll

Makes up a model for an acting PoleCart by pulling all neccessary components to fit into the RT Machine
'''

import datetime as dt
from realtimemachine.classes import State, Prime, Result

class ReforceModel(object):
    """ rather shallow collection that wraps around a scoring function making it into a model
    
    Parameters
    ----------
    name : string
        name of the model
    modelid : int
        an id for the model
    
    Properties
    ----------
    stateBuilders : list of StateBuilders
        the list of statebuilders
    primeBuilders : list of PrimeBuilders
        the list of primeBuilders
    actor : object
        An actor, which takes states and proposes actions
    creationDate : datetime
        the time of this models creation
    trainingDate : datetime
        the date this model has been trained
    trainingId : int
        an id for the training
    valid : bool
        is this Model valid?
    valiationDate : datetime
        the date this model has been validated
    meta : dict
        meta information about the model
    """
    def __init__(self, name, modelId):
        """ configure with minimal information """
        self.name = name
        self.modelId = modelId
        #-------------------
        self.actor = None
        self.primeBuilders = None
        self.stateBuilders = None
        #======================
        self.creationDate = dt.datetime.now()
        self.trainingDate = None
        #----------------
        self.valid = False
        self.valiationDate = None
        #------------------
        self.meta = {}
    def getPrimeBuilders(self):    
        return self.primeBuilders
    def getStateBuilders(self):
        return self.stateBuilders        
    def score(self, prime):
        action_values = self.pcagent.actions_prob(prime.data)
        r = Result(prime.uid, prime.targetid, dt.datetime.now(), prime.routingkey, prime.meta)
        r.results = action_values
        return  r        
    def prepare(self):
        self.actor.prepare()
        return self
    