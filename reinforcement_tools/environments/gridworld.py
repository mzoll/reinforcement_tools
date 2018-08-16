import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt

from skimage.transform import resize as imgresize

class GameEnv():
    """ An gridworld environemnt with a (blue) hero, (red) fire pits and (green) goals.
    Navigate the game grid by moving up, down, left or right and collect as many goals as you can
    manage in order to save rewards.
    
    Parameters
    ----------
    partial : bool
        ???
    size : int >0
        size of the square game-field.
    ngoals : int > 0
        number of tiles which are goals and pay reward of +1 on the gamefield.
    nfire : int > 0
        number of tiles which are fire pits and pay reward of -1 on the gamefield.
    boundary_penalty : float (default: 0.1)
        the (positively defined) reward received if colliding with the boundary (set to negative number to penalize).
    step_penalty : float (default: 0.1)
        the (positively defined) reward received for each step taken (set to negative number to penalize).
    rendersize : int >0
        the size of the rendered gamefield.
    """
    actions = 4
    class _GameOb():
        """ a game observable object """
        def __init__(self,coordinates,size,intensity,channel,reward,name):
            self.x = coordinates[0]
            self.y = coordinates[1]
            self.size = size
            self.intensity = intensity
            self.channel = channel
            self.reward = reward
            self.name = name

    def __init__(self, partial, size, ngoals=3, nfire=2, boundary_penalty=0.1, step_penalty=0.1, render_size=84):
        self.sizeX = size
        self.sizeY = size
        self._ngoals = ngoals
        self._nfire = nfire
        self.partial = partial
        self.boundary_penalty = boundary_penalty
        self.step_penalty = step_penalty
        self.rendersizeX = render_size
        self.rendersizeY = render_size
        self._objects = []
        
        self._hero = None
        self.reset()
    def _create_gameObj(self, otype):
        if otype=='hero':   
            return self._GameOb(self._newPosition(),1,1,2,None,'hero')
        elif otype=='goal':
            return self._GameOb(self._newPosition(),1,1,1,1,'goal')
        elif otype=='fire':
            return self._GameOb(self._newPosition(),1,1,0,-1,'fire')
        else:
            raise Exception('unknown game environment object')
    def reset(self):
        self._hero = self._create_gameObj('hero')
        self._objects = []
        for _ in range(self._ngoals):
            self._objects.append( self._create_gameObj('goal') )
        for _ in range(self._nfire):
            self._objects.append( self._create_gameObj('fire') )
        obs = self._renderEnv()
        self.obs = obs
        return obs

    def _moveChar(self, direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        if direction == 0:
            to_move = (0,-1)
        elif direction == 1:
            to_move = (0,+1)
        elif direction == 2:
            to_move = (-1, 0)
        elif direction == 3:
            to_move = (1,0)
        else:
            raise Exception('illegal move')     
        #check boundary
        valid_move = (0 <= (self._hero.x+to_move[0]) <= (self.sizeX-1)) and (0 <= (self._hero.y+to_move[1]) <= (self.sizeY-1))
        
        if valid_move:
            penalize = False
            self._hero.x += to_move[0]
            self._hero.y += to_move[1]
        else:
            #print('Collision with boundary')
            penalize = True
        return penalize
    
    def _newPosition(self):
        #list all possible positions #fixme, make set
        gridPositions = [] 
        iterables = [ range(self.sizeX), range(self.sizeY)]
        for t in itertools.product(*iterables):
            gridPositions.append(t)
        #list the taken up positions    
        takenPositions = []
        if self._hero is not None:
            takenPositions.append( (self._hero.x, self._hero.y) )
        for objectA in self._objects:
            takenPositions.append((objectA.x, objectA.y))
        #shoot down the taken positions
        for pos in takenPositions:
            gridPositions.remove(pos)

        location = random.choice( gridPositions )
        return location

    def _checkGoal(self):
        ended = False
        for gobj in self._objects:
            if self._hero.x == gobj.x and self._hero.y == gobj.y:
                reward = gobj.reward
                self._objects.remove(gobj)
                #spawn a new object in the world of the same type
                self._objects.append( self._create_gameObj( gobj.name ) )
                return reward,False
        if not ended:
            return 0.0, False

    def _renderEnv(self):
        #black gamefield, white boarders, 
        a = np.zeros([self.sizeY+2,self.sizeX+2,3])
        a[0,:,:] = 1
        a[self.sizeY+1,:,:] = 1
        a[:,0,:] = 1
        a[:,self.sizeX+1,:] = 1
        
        #a = np.ones([self.sizeY+2,self.sizeX+2,3])
        #a[1:-1,1:-1,:] = 0
        for item in self._objects:
            a[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,item.channel] = item.intensity
        item = self._hero
        a[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,item.channel] = item.intensity
        #what does this do?
        if self.partial:
            a = a[self._hero.y:self._hero.y+3,self._hero.x:self._hero.x+3,:]
        
        obs = imgresize(a, (self.rendersizeX,self.rendersizeY,3), mode='constant', order=0)
        return obs

    def step(self, action):
        """ perform one step in the environment
        
        Parameters
        ----------
        action : int in [0..3]
            the action to perform
            
        Returns
        -------
        tuple of (obs, reward, done)
        """
        penalty = self._moveChar(action)
        reward,done = self._checkGoal()
        obs = self._renderEnv()
        return obs, (reward + penalty*self.boundary_penalty + self.step_penalty), done
    @property
    def state(self):
        """ the current observable state of the environment """
        return self._renderEnv()