'''
Created on Jul 20, 2018

@author: marcel.zoll
'''

from reinforcement.actor import ReforceActor

class PoleCartActor(ReforceActor):
    """ A derivative of the Reforce Agent for the PoleCart learning example
    """
    _env_name = 'CartPole-v0'
    _state_vars = ['0_pos_cart', '1_vel_cart', '2_angle_pole', '3_rotrate_pole']
    _action_space = ['left', 'right']
    
    def __init__(self):
        ReforceActor.__init__(self, 
            state_size = len(self._state_vars),
            action_size = len(self._action_space),
            hidden_size = 8,
            learning_rate = 1e-2)
    
    def actions_prob(self, var_dict):
        pcstate = [ var_dict[k] for k in self._state_vars ]
        a_vec = self.prob_actions(pcstate)
        return dict(zip(self._action_space, a_vec))