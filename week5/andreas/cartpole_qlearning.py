import gym
import random
import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class QSolver():
    def __init__(self, env, eps, alpha, gamma, batch_size=5000, memory_size=1000000):
        self.env = env
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(
            self.env.observation_space.shape[0],), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.env.action_space.n, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001))

#         # Random Forest
#         self.model = RandomForestRegressor()

#         # Initiate Model
#         X = np.random.random_sample((100, 5))
#         y = np.repeat(1, 100)
#         self.model.fit(X, y)

        # Linear Model
        #self.model = LinearRegression()

        # Initiate Model Parameters
        #self.model.intercept_ = 0
        #self.model.coef_ = np.zeros(self.env.observation_space.shape[0] + 1)

    def remember(self, state, action, newReward, newState, terminated):
        self.memory.append((state, action, newReward, newState, terminated))

#     def feature_matrix(self, state):
#         return np.array([list(state) + [0],
#                         list(state) + [1]])

    def greedy_action(self, state):
        #         X = self.feature_matrix(state)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def epsilon_greedy(self, state):
        if np.random.random() >= self.eps:
            return self.greedy_action(state)

        return self.env.action_space.sample()

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
#         target_q_values = []
#         X_list = []
        for state, action, newReward, newState, terminated in batch:
            q_update = newReward
            if not terminated:
                #                 X_next = self.feature_matrix(newState)
                q_update = (newReward + self.gamma *
                            np.amax(self.model.predict(newState)[0]))
#             x = self.feature_matrix(state)[action]
#             X_list.append(x)
            q_values = self.model.predict(state)
            q_values[0][action] = (1-self.alpha) * \
                q_values[0][action] + self.alpha*q_update
            #print(f"state {np.array(state).shape}")
            #print(f"q_value: {q_values}")

#             target_q_values.append(q_value)
            self.model.fit(state, q_values, verbose=0)

#         X = np.array(X_list)
#         target_q_values = np.array(target_q_values)

        # Updating value function parameters
        #print(f"X.shape: {X.shape}, target_q_values.shape: {target_q_values.shape}")
#         self.model.fit(X, target_q_values)

    def step(self, action):
        return self.env.step(action)

    def run_episode(self, render=False, exposition=False):
        state = self.env.reset()
        state = np.reshape(state, [1, self.env.observation_space.shape[0]])

        t = 0
        while True:
            t += 1

            if render:
                self.env.render()

            if exposition:
                action = self.greedy_action(state)
            else:
                action = self.epsilon_greedy(state)

            newState, newReward, terminated, info = self.env.step(action)
            newState = np.reshape(
                newState, [1, self.env.observation_space.shape[0]])

            if not exposition:
                # Add to memory
                self.remember(state, action, newReward, newState, terminated)

                # Update Model
                self.experience_replay()

#             if not exposition:
#                 self.update_Q(state, action, newReward, newState)

            # Update state
            #print(f"state {state}")
            #print(f"newState {newState}")
            state = newState

            if terminated:
                print(f"Episode length: {t}")
                break

        self.env.close()

    def run_q_learning(self, n_episodes, render=False):

        for t in range(n_episodes):
            #            old_params = self.model.coef_.copy()
            print(f"Episode: {t+1}")
            self.run_episode()

#             if len(self.memory) > self.batch_size and np.allclose(old_params, self.model.coef_,  0.05):
#                 print(f'Learning Converged. Parameters: {self.model.coef_}')
#                 break

#            self.eps = 1/(t+1)


env = gym.make('CartPole-v1')
env.reset()

eps = 0.5
alpha = 0.05
gamma = 0.95
solver = QSolver(env, eps, alpha, gamma)
solver.run_q_learning(n_episodes=1000)
