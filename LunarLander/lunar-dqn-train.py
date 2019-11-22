import numpy as np
import gym
#import matplotlib.pyplot as plt


import sys
import os.path

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import Callback

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'LunarLander-v2'

nb_steps = int(sys.argv[1])

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
#np.random.seed(123)
#env.seed(123)
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
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, enable_dueling_network=True, )
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

fname = f'dqn_{ENV_NAME}_weights.h5f'.lower()
if os.path.isfile(fname):
    dqn.load_weights(fname)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

#class Metrics(Callback):
#    def __init__(self, agent):
#        Callback.__init__(self)
#        self.agent = agent
#
#    def on_train_begin(self, logs={}):
#        self.metrics = {key : [] for key in self.agent.metrics_names}
#
#    def on_step_end(self, episode_step, logs):
#        for ordinal, key in enumerate(self.agent.metrics_names, 0):
#            self.metrics[key].append(logs.get('metrics')[ordinal])
#metrics = Metrics(dqn)

#dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=2, callbacks=[metrics])
dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights(fname, overwrite=True)

#print(metrics.metrics.keys())
#plt.plot(metrics.metrics['mean_absolute_error'])
#plt.title('MAE')
#plt.ylabel('mae')
#plt.xlabel('step')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

print('Done')

#input("Hit enter to test trained agent...")

# Finally, evaluate our algorithm for 5 episodes.
#dqn.test(env, nb_episodes=5, visualize=True)

# Finally, evaluate our algorithm for 5 episodes.
#dqn.load_weights(f'dqn_{ENV_NAME}_weights.h5f')
#dqn.test(env, nb_episodes=5, visualize=False)
