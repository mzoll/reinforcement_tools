import numpy as np
import time
from gridworld import GameEnv
import matplotlib.pyplot as plt

env= GameEnv(partial=False, size=5)

state= env.reset()
print(state)

for i in range(500):
    a = np.random.randint(4)
    state = env.step(a)
    plt.imshow(state)
    plt.draw()
time.sleep(5)

raise Exception()


from reinforcement.episode import Episode, Transition,EpisodesBuffer

b = EpisodesBuffer()

def nstate():
    return np.random.random_integers(0, 10, (2,2))

state = nstate()

for i in range(3):
    epi = Episode(state)
    for j in range(2):
        next_state = nstate()
        epi.addTransition( Transition(state, np.random.randint(0,4), 0.5, next_state) )
        state = next_state
    b.add(epi)
    
print(epi.getArrays())

print('***')

print( b.getArrays())
