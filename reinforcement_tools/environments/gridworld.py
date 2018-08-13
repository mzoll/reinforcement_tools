import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt

from skimage.transform import resize as imgresize

class GameEnv():
    class GameOb():
        """ a game observable """
        def __init__(self,coordinates,size,intensity,channel,reward,name):
            self.x = coordinates[0]
            self.y = coordinates[1]
            self.size = size
            self.intensity = intensity
            self.channel = channel
            self.reward = reward
            self.name = name

    def __init__(self,partial, size, ngoals=3, nfire=2):
        self.sizeX = size
        self.sizeY = size
        self._ngoals = ngoals
        self._nfire = nfire
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.boundary_penalty = 0.
        
        self.hero = None
        self.reset()
    def create_gameObj(self, otype):
        if otype=='hero':   
            return self.GameOb(self.newPosition(),1,1,2,None,'hero')
        elif otype=='goal':
            return self.GameOb(self.newPosition(),1,1,1,1,'goal')
        elif otype=='fire':
            return self.GameOb(self.newPosition(),1,1,0,-1,'fire')
        else:
            raise Exception('unknown game environment object')
    def reset(self):
        self.hero = self.create_gameObj('hero')
        self.objects = []
        for _ in range(self._ngoals):
            self.objects.append( self.create_gameObj('goal') )
        for _ in range(self._nfire):
            self.objects.append( self.create_gameObj('fire') )
        obs = self.renderEnv()
        self.obs = obs
        return obs

    def moveChar(self, direction):
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
        valid_move = (0 <= (self.hero.x+to_move[0]) <= (self.sizeX-1)) and (0 <= (self.hero.y+to_move[1]) <= (self.sizeY-1))
        
        if valid_move:
            penalize = 0.
            self.hero.x += to_move[0]
            self.hero.y += to_move[1]
        else:
            #print('Collition with boundary')
            penalize = self.boundary_penalty
        return penalize
    
    def newPosition(self):
        #list all possible positions #fixme, make set
        gridPositions = [] 
        iterables = [ range(self.sizeX), range(self.sizeY)]
        for t in itertools.product(*iterables):
            gridPositions.append(t)
        #list the taken up positions    
        takenPositions = []
        if self.hero is not None:
            takenPositions.append( (self.hero.x, self.hero.y) )
        for objectA in self.objects:
            takenPositions.append((objectA.x, objectA.y))
        #shoot down the taken positions
        for pos in takenPositions:
            gridPositions.remove(pos)

        location = random.choice( gridPositions )
        return location

    def checkGoal(self):
        ended = False
        for gobj in self.objects:
            if self.hero.x == gobj.x and self.hero.y == gobj.y:
                reward = gobj.reward
                self.objects.remove(gobj)
                #spawn a new object in the world of the same type
                self.objects.append( self.create_gameObj( gobj.name ) )
                return reward,False
        if not ended:
            return 0.0, False

    def renderEnv(self):
        #black gamefield, white boarders, 
        a = np.zeros([self.sizeY+2,self.sizeX+2,3])
        a[0,:,:] = 1
        a[self.sizeY+1,:,:] = 1
        a[:,0,:] = 1
        a[:,self.sizeX+1,:] = 1
        
        #a = np.ones([self.sizeY+2,self.sizeX+2,3])
        #a[1:-1,1:-1,:] = 0
        for item in self.objects:
            a[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,item.channel] = item.intensity
        item = self.hero
        a[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,item.channel] = item.intensity
        #what does this do?
        if self.partial:
            a = a[self.hero.y:self.hero.y+3,self.hero.x:self.hero.x+3,:]
        
        obs = imgresize(a, (84,84,3), mode='constant', order=0)
        return obs

    def step(self,action):
        penalty = self.moveChar(action)
        reward,done = self.checkGoal()
        obs = self.renderEnv()
        return obs,(reward+penalty),done
    @property
    def state(self):
        """ copatibility measure """
        return self.renderEnv()