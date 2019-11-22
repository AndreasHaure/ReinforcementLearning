import numpy as np
import gym
from time import sleep
import os.path

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

SEED = 4
ENV_NAME = 'LunarLander-v2'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(SEED)
env.seed(SEED)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
#print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, enable_dueling_network=True)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.test(env, nb_episodes=5, visualize=True)

for i in range(3,0,-1):
    print('Sentient agent in', i)
    sleep(1)

fname = f'dqn_{ENV_NAME}_weights.h5f'.lower()
if not os.path.isfile(fname):
    fname = f'dqn_{ENV_NAME}_weights_20190321.h5f'.lower()

dqn.load_weights(fname)
dqn.test(env, nb_episodes=5, visualize=True)
