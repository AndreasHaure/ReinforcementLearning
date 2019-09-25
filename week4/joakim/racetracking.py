from numpy import genfromtxt
import numpy as np
import os
from collections import defaultdict
from itertools import permutations, repeat
import random
from numpy import random as numpy_random

# Load map1
CELL_TYPE_WALL = 0  # Black boxes
CELL_TYPE_TRACK = 1
CELL_TYPE_GOAL = 2
CELL_TYPE_START = 3


class RaceTrack:
    def __init__(self, track, zero_velocity_prob=0.9, max_vel=5, min_vel=0, gamma=0.95, epsilon=0.95):
        self.track = track
        self.wall_cells = np.argwhere(track == CELL_TYPE_WALL)
        self.goal_cells = np.argwhere(track == CELL_TYPE_GOAL)
        self.start_cells = np.argwhere(track == CELL_TYPE_START)
        self.max_vel = max_vel
        self.min_vel = min_vel
        self.colors = ['black', 'white', 'yellow', 'red']  # For plotting
        self.gamma = gamma
        self.epsilon = epsilon

        self.zero_velocity_prob = zero_velocity_prob
        self.velocity_min = min_vel
        self.velocity_max = max_vel
        self.velocity_decrease_limit = -1
        self.velocity_increase_limit = 1

        # Q-Matrix - a 6 dimensional vector for states, velocity, and action in both directions.

        # Q(s, a)
        # Minus for other direction.
        self.velocity_range = self.velocity_max - self.velocity_min + 1
        self.velocity_change_range = self.velocity_increase_limit - \
            self.velocity_decrease_limit + 1
        self.q = np.zeros((self.track.shape[0], self.track.shape[1], self.velocity_range,
                           self.velocity_range, self.velocity_change_range, self.velocity_change_range))

        # Returns
        self.Returns = defaultdict(list)

        # Initialize with equal probabilties for all possible actions
        self.update_policy(initialize=True)

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
                self.Returns[s_a].append(G[s_a])

            print('\nOld Q-values ')
            print('Q-value  [0, 8, 0, 0, 1, 0]:', self.q[0, 8, 0, 0, 1, 0])
            print('Q-value  [0, 8, 0, 0, 1, 1]:', self.q[0, 8, 0, 0, 1, 1])
            print('Q-value  [0, 8, 0, 0, 0, 1]:', self.q[0, 8, 0, 0, 0, 1])

            # Replace q-values with average returns.
            for s_a in self.Returns.keys():
                self.q[eval(s_a)[0],
                       eval(s_a)[1],
                       eval(s_a)[2],
                       eval(s_a)[3],
                       eval(s_a)[4],
                       eval(s_a)[5]] = np.average(self.Returns[s_a])
            print('\nUpdated Q-values')
            print('Q-value  [0, 8, 0, 0, 1, 0]:', self.q[0, 8, 0, 0, 1, 0])
            print('Q-value  [0, 8, 0, 0, 1, 1]:', self.q[0, 8, 0, 0, 1, 1])
            print('Q-value  [0, 8, 0, 0, 0, 1]:', self.q[0, 8, 0, 0, 0, 1])
            print('\n')

            # Old policy
            old_policy = self.pi_probabilities.copy()

            # Update pi(a | s)
            self.update_policy(initialize=False)

            # Check if convergence
            if np.allclose(old_policy, self.pi_probabilities, atol=0.0005):
                print('Policy iteration converged.')
                policy_improvement = True

            # Counter and update epsilon
            self.epsilon = 1/(np.sqrt(k + 1.1))

            if k > 150:
                print('ok...')
            k += 1

    def generate_episode(self):
        """
        """

        crossed_finishing_line = False
        position = self.random_start_position()
        first_occurence = defaultdict(int)
#        all_remaining_positions = []

        step = 0
        while not crossed_finishing_line:

            # Sample action
            action = self.sample_action_from_state(position)

            # Initiate s, a pair if not already in dict
            if str(position + action) not in first_occurence.keys():
                first_occurence[str(position + action)] = step
                if str(position + action) == '[0, 8, 0, 0, 1, 0]':
                    print('Visited:', '[0, 8, 0, 0, 1, 0]', 'in step', step)
#                    all_remaining_positions = []
                if str(position + action) == '[0, 8, 0, 0, 1, 1]':
                    print('Visited:', '[0, 8, 0, 0, 1, 1]', 'in step', step)
                if str(position + action) == '[0, 8, 0, 0, 0, 1]':
                    print('Visited:', '[0, 8, 0, 0, 0, 1]', 'in step', step)

#            all_remaining_positions.append(position)

            # Old position
            old_position = position.copy()

            # New position
            position[0] += position[2] + action[0]
            position[1] += position[3] + action[1]
            position[2] += action[0]
            position[3] += action[1]

            # Update step
            step += 1

            # Check if goal if reached (is it in the projected reactangle)
            grid_states_to_check = self.get_all_grid_cells_in_projected_retcangle(current_state=[old_position[0], old_position[1]],
                                                                                  new_state=[position[0], position[1]])
            if self.check_if_goal_is_reached(check_grid_states=grid_states_to_check):
                print('-- Goal Reached. Terminating Episode.')
                break

            # Check if car hits boundery
            if position[0] >= self.track.shape[0] or position[1] >= self.track.shape[1]:
                position = self.random_start_position()
#                print('Outside track cells!')
                continue

            # Check if car is in is black cells (outside track)
            new_grid_position = rt.track[position[0], position[1]]
            if new_grid_position == 0:
                #                print('Car hit track boundery!')
                position = self.random_start_position()
                continue

        print('Steps {}'.format(step))
#        print(all_remaining_positions)

        G = self._get_G_values(
            first_occurence_dict=first_occurence, total_steps=step)

        return G

    def check_if_goal_is_reached(self, check_grid_states):
        """
        """

        grid_values = []

        for y, x in check_grid_states:
            if y <= self.track.shape[0] - 1 and x <= self.track.shape[1] - 1:
                grid_values.append(rt.track[y, x])

        # Goal cell type
        if 2 in grid_values:
            return True
        else:
            return False

    def get_all_grid_cells_in_projected_retcangle(self, current_state, new_state):
        """
        """
        y_coord_current = current_state[0]
        x_coord_current = current_state[1]

        y_diff = new_state[0] - current_state[0]
        x_diff = new_state[1] - current_state[1]

        return [[y_coord_current + y, x_coord_current + x] for y in range(0, y_diff + 1) for x in range(0, x_diff + 1)]

    def sample_action_from_state(self, state):
        """
        """

        # Action coordinates in probability matrix
        array = [[0, 0],
                 [0, 1],
                 [0, 2],
                 [1, 0],
                 [1, 1],
                 [1, 2],
                 [2, 0],
                 [2, 1],
                 [2, 2]]

        # Randomly pick action
        idx = numpy_random.choice(range(9),
                                  size=1,
                                  p=rt.pi_probabilities[state[0], state[1], state[2], state[3]].flatten())

        # Translate back to range -1, 1 and return result
        return [acc if acc <= 1 else 1 - acc for acc in array[idx[0]]]

    def greedy_action(self, state, possible_actions):
        """
        """

        greedy_action = possible_actions[0]
        q_max = self.q[state[0],
                       state[1],
                       state[2],
                       state[3],
                       greedy_action[0],
                       greedy_action[1]]

        for action in possible_actions:
            value = self.q[state[0], state[1], state[2],
                           state[3], action[0], action[1]]
            if value > q_max:
                q_max = value
                greedy_action = action

        # Translate back to range -1, 1 and return result
        return [acc if acc <= 1 else 1 - acc for acc in greedy_action]

    def epsilon_soft_policy(self, action, greedy_action, all_state_actions):
        """
        """
        if action == greedy_action:
            return 1 - self.epsilon + self.epsilon/len(all_state_actions)

        return self.epsilon/len(all_state_actions)

    def random_start_position(self):
        """
        """

        grid_position = list(random.choice(np.argwhere(self.track == 3)))
        velocity = [0, 0]

        return grid_position + velocity

    def possible_actions(self, velocity):
        """
        Credit: Andreas
        """
        actions = [[a_y, a_x] for a_y in range(-1, 2) for a_x in range(-1, 2)]
        legal_actions = []

        v_y, v_x = velocity

        # Discard illegal actions
        for a in actions:
            a_y, a_x = a
            # Cannot go above speed limit in any x direction
            if v_x + a_x < self.min_vel or v_x + a_x > self.max_vel:
                continue
            # Cannot go above speed limit in any y direction
            if v_y + a_y < self.min_vel or v_y + a_y > self.max_vel:
                continue
            # Cannot noop
            if v_x + a_x == 0 and v_y + a_y == 0:
                continue
            legal_actions.append(a)

        return legal_actions

    def _get_G_values(self, first_occurence_dict, total_steps):
        """
        """

        # Dict. w/ G for first occurence for each s, a pair.
        G = defaultdict(int)

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

        self.pi_probabilities = self.initialize_empty_probabilities(
            track=self.track,
            velocity_range=self.velocity_range,
            velocity_change_range=self.velocity_change_range)

        # Initialize with equal probabilties for all possible actions
        for y_coord in range(self.pi_probabilities.shape[0]):
            for x_coord in range(self.pi_probabilities.shape[1]):
                for y_vel in range(self.velocity_min, self.velocity_max + 1):
                    for x_vel in range(self.velocity_min, self.velocity_max + 1):
                        possible_actions = self.possible_actions(
                            (y_vel, x_vel))
                        if initialize:
                            for y_vel_change, x_vel_change in possible_actions:
                                self.pi_probabilities[y_coord,
                                                      x_coord,
                                                      y_vel,
                                                      x_vel,
                                                      y_vel_change,
                                                      x_vel_change] = 1/len(possible_actions)
                        else:
                            greedy_action = self.greedy_action(
                                state=[y_coord, x_coord, y_vel, x_vel],
                                possible_actions=possible_actions)
                            for y_vel_change, x_vel_change in possible_actions:
                                self.pi_probabilities[y_coord,
                                                      x_coord,
                                                      y_vel,
                                                      x_vel,
                                                      y_vel_change,
                                                      x_vel_change] = self.epsilon_soft_policy(action=[y_vel_change, x_vel_change],
                                                                                               greedy_action=greedy_action,
                                                                                               all_state_actions=possible_actions)

    def initialize_empty_probabilities(self, track, velocity_range, velocity_change_range):
        """
        """
        # Initialize policy
        pi_probabilities = np.zeros((track.shape[0],
                                     track.shape[1],
                                     velocity_range,
                                     velocity_range,
                                     velocity_change_range,
                                     velocity_change_range), dtype=float)

        return pi_probabilities

    @classmethod
    def from_csv(cls, file_path):

        file_path = os.path.join(os.getcwd(), file_path)

        track = genfromtxt(file_path, delimiter=',')
        track = np.flip(track, axis=0)

        return cls(track)


# Run agent
rt = RaceTrack.from_csv("../racetracks/map1.csv")
rt.policy_iteration()
