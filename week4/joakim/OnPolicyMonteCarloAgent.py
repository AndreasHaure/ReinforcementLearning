from numpy import genfromtxt
import numpy as np
import os
from collections import defaultdict
from itertools import permutations, repeat
import random
import math
import sys
from numpy import random as numpy_random
from Racetrack import RaceTrack


class OnPolicyMonteCarloAgent:
    def __init__(self, track, n_episodes=10000, gamma=0.9, epsilon=0.9):

        self.track = track
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.epsilon = epsilon

        # Initialize Q values and C values
        y_range = track.track.shape[0]
        x_range = track.track.shape[1]
        self.yvel_range = track.max_vel - track.min_vel + 1
        self.xvel_range = track.max_vel - track.min_vel + 1
        self.yacc_range = 3  # -1, 0, +1
        self.xacc_range = 3  # -1, 0, +1

        # Initialize state-action values
        self.Q = np.zeros((y_range, x_range, self.yvel_range,
                           self.xvel_range, self.yacc_range, self.xacc_range))

        # Initialize rewards dictionary
        self.Returns = {}

        # Initial Policy
        # For each state: assign equal probability of selecting each valid action from the state
        self.pi = np.zeros(self.Q.shape, dtype=float)
        for y_coord in range(self.Q.shape[0]):
            for x_coord in range(self.Q.shape[1]):
                for y_vel in range(track.min_vel, track.max_vel + 1):
                    for x_vel in range(track.min_vel, track.max_vel + 1):
                        valid_actions = self.track.possible_actions(
                            (y_coord, x_coord, y_vel, x_vel))
                        for y_acc, x_acc in valid_actions:
                            self.pi[y_coord, x_coord, y_vel, x_vel,
                                    y_acc, x_acc] = 1/len(valid_actions)

    def policy_iteration(self):
        """
        """

        policy_improvement = False

        k = 0
        while not policy_improvement:
            print('Iteration {}'.format(k))

            # Generate an episode
            G = self.generate_episode()

            # Append G  values to Returns
            for s_a in G.keys():
                # Create key if not in Returns
                if s_a not in self.Returns.keys():
                    self.Returns[s_a] = []
                self.Returns[s_a].append(G[s_a])

            # Replace q-values with average returns.
            for s_a in self.Returns.keys():
                self.Q[eval(s_a)[0],
                       eval(s_a)[1],
                       eval(s_a)[2],
                       eval(s_a)[3],
                       eval(s_a)[4],
                       eval(s_a)[5]] = np.average(self.Returns[s_a])
            print('\nUpdated Q-values')
            print('Q-value  [0, 8, 0, 0, 1, 0]:', self.Q[0, 8, 0, 0, 1, 0])
            print('Q-value  [0, 8, 0, 0, 1, 1]:', self.Q[0, 8, 0, 0, 1, 1])
            print('Q-value  [0, 8, 0, 0, 0, 1]:', self.Q[0, 8, 0, 0, 0, 1])
            print('\n')

            # Old policy
            old_policy = self.pi.copy()

            # Update pi(a | s)
            self.update_policy(initialize=False)

            # Check if convergence
            if np.allclose(old_policy, self.pi, atol=0.0001):
                print('Policy iteration converged.')
                policy_improvement = True

            # Counter and update epsilon
            self.epsilon = 1/(np.sqrt(k + 1.1))

            k += 1

    def generate_episode(self):
        """
        """

        crossed_finishing_line = False
        position = list(self.track.random_start_state())
        first_occurence = defaultdict(int)

        step = 0
        while not crossed_finishing_line:

            # Sample action
            action = self.sample_action_from_state(position)

            # Initiate s, a pair if not already in dict
            if str(position + action) not in first_occurence.keys():
                first_occurence[str(position + action)] = step

            # New position
            position = list(self.track.apply_action(
                state=position, action=action)[0])

            # Update step
            step += 1

            # Get projected path
            projected_path = self.track.projected_path(
                state=[position[0], position[1]], speed=[position[2], position[3]])

            # Check if goal if reached (is it in the projected reactangle)
            if self.track.crossed_finish_line(projected_path=projected_path):
                print('-- Goal Reached. Terminating Episode.')
                break

            # Check if car hits boundery or wall cells
            if self.track.crossed_track_boundary(projected_path=projected_path):
                position = self.random_start_position()
#                print('Outside track cells!')
                continue

        print('Steps {}'.format(step))

        G = self._get_G_values(
            first_occurence_dict=first_occurence, total_steps=step)

        return G

    def sample_action_from_state(self, state):
        """
        """

        # Sample action according to our eps-greedy policy
        # Ensure that probabilities we sample from sum to 1
        y_coord, x_coord, y_vel, x_vel = state

        actionprobs = self.pi[y_coord, x_coord, y_vel, x_vel]
        total_prob = np.sum(actionprobs)
        if not math.isclose(total_prob, 1, abs_tol=0.01):
            print(
                'Action probabilities must sum to 1.0, but summed to {}, state: {}, actionprobs: {}'.format(total_prob, state, self.pi[tuple(state)]))
            sys.exit(1)

        linear_idx = np.random.choice(
            actionprobs.size, p=actionprobs.ravel())
        a = np.unravel_index(linear_idx, actionprobs.shape)
        # In case the value is greater than the max allowed action we need to translate it back into
        # negative coordinates
        a = [acc if acc <= 1 else 1 - acc for acc in a]

        return a

    def greedy_action(self, state):

        # Find greedy action according to state-action values Q
        Q_state = self.Q[tuple(state)].copy()
        if not (Q_state == 0).all():
            Q_state[Q_state == 0] = np.nan
        a = np.unravel_index(np.nanargmax(Q_state, axis=None), Q_state.shape)
        # In case the value is greater than the max allowed action we need to translate it back into
        # negative coordinates
        a = [acc if acc <= 1 else 1 - acc for acc in a]

        return a

    def epsilon_soft_policy(self, action, greedy_action, all_state_actions):
        """
        """
        if greedy_action:
            if action == greedy_action:
                return 1 - self.epsilon + self.epsilon/len(all_state_actions)

            return self.epsilon/len(all_state_actions)

        return 1/len(all_state_actions)

    def random_start_position(self):
        """
        """

        grid_position = list(random.choice(self.track.start_cells))
        velocity = [0, 0]

        return grid_position + velocity

    def _get_G_values(self, first_occurence_dict, total_steps):
        """
        """

        # Dict. w/ G for first occurence for each s, a pair.
        G = {}

        for key, val in first_occurence_dict.items():
            number_rewards = total_steps - val

            discounted_rewards = []
            for k in range(number_rewards):
                discounted_rewards.append(self.gamma**k * (-1))

            G[key] = sum(discounted_rewards)

        return G

    def update_policy(self, initialize):
        """
        """

        # Initialize with equal probabilties for all possible actions
        for y_coord in range(self.pi.shape[0]):
            for x_coord in range(self.pi.shape[1]):
                for y_vel in range(self.xvel_range):
                    for x_vel in range(self.xvel_range):
                        possible_actions = self.track.possible_actions(
                            [y_coord, x_coord, y_vel, x_vel])
                        for a in possible_actions:

                            self.pi[y_coord, x_coord, y_vel, x_vel, a[0], a[1]] = self.epsilon / \
                                len(possible_actions)

                        # Get index of best action
                        a_ys, a_xs = tuple(zip(*possible_actions))
                        actionvals = self.Q[y_coord, x_coord, y_vel,
                                            x_vel, a_ys, a_xs]
                        a_max_idx = np.argmax(actionvals)
                        a_max_y, a_max_x = a_ys[a_max_idx], a_xs[a_max_idx]

                        self.pi[y_coord, x_coord, y_vel, x_vel, a_max_y,
                                a_max_x] += 1 - self.epsilon

                        actionprobs = self.pi[y_coord,
                                              x_coord, y_vel, x_vel]
                        total_prob = np.sum(actionprobs)
                        if not math.isclose(total_prob, 1, abs_tol=0.01):
                            print(
                                'Action probabilities must sum to 1.0, but summed to {}, state: {}, actionprobs: {}'.format(total_prob, [y_coord, x_coord, y_vel, x_vel], self.pi[y_coord, x_coord, y_vel, x_vel]))
                            sys.exit(1)

    @classmethod
    def from_csv(cls, file_path):

        file_path = os.path.join(os.getcwd(), file_path)

        track = genfromtxt(file_path, delimiter=',')
        track = np.flip(track, axis=0)

        return cls(track)


# Load map
rt = RaceTrack.from_csv("../racetracks/map1.csv")

# Run agent
agent = OnPolicyMonteCarloAgent(rt, epsilon=0.5, gamma=0.9)
agent.policy_iteration()
