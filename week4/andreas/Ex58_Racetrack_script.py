from collections import defaultdict
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import time

CELL_TYPE_WALL = 0
CELL_TYPE_TRACK = 1
CELL_TYPE_GOAL = 2
CELL_TYPE_START = 3


class RaceTrack:
    def __init__(self, track, max_vel=5, min_vel=0):
        self.track = track
        self.wall_cells = np.argwhere(track == CELL_TYPE_WALL).tolist()
        self.goal_cells = np.argwhere(track == CELL_TYPE_GOAL).tolist()
        self.start_cells = np.argwhere(track == CELL_TYPE_START).tolist()
        self.max_vel = max_vel
        self.min_vel = min_vel

    @classmethod
    def from_csv(cls, file_path):

        file_path = os.path.join(os.getcwd(), file_path)

        track = genfromtxt(file_path, delimiter=',')
        # Flip the y-axis coordinates
        track = np.flip(track, axis=0)

        return cls(track)

    def possible_actions(self, state):
        actions = [[a_y, a_x] for a_y in range(-1, 2) for a_x in range(-1, 2)]
        legal_actions = []

        _, _, v_y, v_x = state

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

    def random_start_state(self):
        start_cell_idx = np.random.choice(len(self.start_cells))
        start_state = np.array(self.start_cells[start_cell_idx] + [0, 0])
        return start_state

    def apply_action(self, state, action):
        y_coord, x_coord, y_vel, x_vel = state
        a_y, a_x = action

        next_y_vel = y_vel + a_y
        next_x_vel = x_vel + a_x
        next_y_coord = y_coord + next_y_vel
        next_x_coord = x_coord + next_x_vel

        path = self.projected_path(
            (y_coord, x_coord), (next_y_vel, next_x_vel))

        if self.crossed_finish_line(path):
            return self.random_start_state(), 0, True
        if self.crossed_track_boundary(path):
            # if self.crossed_track_boundary([(next_y_coord, next_x_coord)]):
            return self.random_start_state(), -1, False

        return np.array([next_y_coord, next_x_coord, next_y_vel, next_x_vel]), -1, False

    def projected_path(self, state, speed):
        # TODO: Should we only consider end state directly?
        y_coord, x_coord = state
        y_vel, x_vel = speed

        new_y_coord = y_coord + y_vel
        new_x_coord = x_coord + x_vel

        path = []
        for dy in range(min(y_coord, new_y_coord), max(y_coord, new_y_coord) + 1):
            for dx in range(min(x_coord, new_x_coord), max(x_coord, new_x_coord) + 1):
                path.append([dy, dx])
        return path

    def crossed_track_boundary(self, projected_path):
        for cell in projected_path:
            y, x = cell
            if y < 0 or y >= self.track.shape[0] or x < 0 or x >= self.track.shape[1] or cell in self.wall_cells:
                return True
        return False

    def crossed_finish_line(self, projected_path):
        for cell in projected_path:
            if cell in self.goal_cells:
                return True
        return False

    def draw(self, car_cell=None, path=[]):
        colors = ['black', 'white', 'yellow', 'red']

        fig = plt.figure(figsize=(10, 10), dpi=80,
                         facecolor='w', edgecolor='k')

        im = plt.imshow(self.track, cmap=ListedColormap(colors),
                        origin='lower', interpolation='none', animated=True)

        def rect(pos, edgecolor='k', facecolor='none'):
            r = plt.Rectangle(pos, 1, 1, facecolor=facecolor,
                              edgecolor=edgecolor, linewidth=2)
            plt.gca().add_patch(r)

        for i in range(self.track.shape[0]):
            for j in range(self.track.shape[1]):
                rect((j-0.5, i-0.5))

        if path:
            for cell in path:
                rect((cell[1]-0.5, cell[0]-0.5), edgecolor='g')

        if car_cell:
            rect((car_cell[1]-0.5, car_cell[0]-0.5),
                 edgecolor='g', facecolor='g')

        plt.gca().invert_yaxis()
        return im


class OnPolicyMonteCarloAgent:
    def __init__(self, track, gamma=1, n_episodes=100000, eps=0.1):
        self.track = track
        self.gamma = gamma
        self.n_episodes = n_episodes

        # Initialize Q values and C values
        y_range = track.track.shape[0]
        x_range = track.track.shape[1]
        yvel_range = track.max_vel - track.min_vel + 1
        xvel_range = track.max_vel - track.min_vel + 1
        yacc_range = 3  # -1, 0, +1
        xacc_range = 3  # -1, 0, +1

        # Initialize state-action values
        self.Q = np.zeros((y_range, x_range, yvel_range,
                           xvel_range, yacc_range, xacc_range))

        # Initialize rewards dictionary
        self.R = defaultdict(list)

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

    def sample_random_action(self, state):

        # Sample action according to our eps-greedy policy
        # Ensure that probabilities we sample from sum to 1
        actionprobs = self.pi[tuple(state)]
        assert np.sum(actionprobs), 'Action probabilities must sum to 1.0'

        linear_idx = np.random.choice(
            actionprobs.size, p=actionprobs.ravel())
        a = np.unravel_index(linear_idx, actionprobs.shape)
        # In case the value is greater than the max allowed action we need to translate it back into
        # negative coordinates
        a = tuple(acc if acc <= 1 else 1 - acc for acc in a)
        return a

    def generate_episode(self, pi):
        S = []
        A = []
        R = []

        # Select the initial state randomly
        start_state = self.track.random_start_state()
        S.append(start_state)

        terminated = False
        t = 0
        while not terminated:
            St = S[t]
            print("Step: {}, {}".format(t, St))
            a = self.sample_random_action(St)
            A.append(a)

            next_state, reward, terminated = self.track.apply_action(St, a)

            R.append(reward)
            S.append(next_state)
            t += 1
        return S, A, R

    def solve_track(self):
        it = 0
        while it < self.n_episodes:
            # (a) Generate an episode using pi
            S, A, R = self.generate_episode(self.pi)
            print(S)
            return

            # (b) Iterate over s,a pairs and update rewards and q-values
            G_first_visit = {}
            for t in range(0, len(S)):
                # If first time visit in state S taking action A:
                St, At, Rt = S[t], A[t], R[t]

                state_action_key = tuple(St + At)
                if (St, At) not in G_first_visit:
                    G_first[(St, At)] = Rt
                    self.R[tuple(state_action_key)].append(Rt)
                    self.Q[tuple(stat_action_key)] = np.average(
                        self.R[tuple(state_action_key)])

            # (c) Iterate over all states s and update the eps-greedy policy
            for t in range(S):
                St = S[t]
                y, x, y_vel, x_vel = St
                actionvals = self.Q[y, x, y_vel, x_vel]

                # Get index of best action. However this is in positive only coordinates.
                # In case the value is greater than the max allowed action we need to translate it back into
                # negative coordinates
                a_max = np.unravel_index(actionvals.argmax(), actionvals.shape)
                a_max = tuple(acc if acc <= 1 else 1 - acc for acc in a_max)

                possible_actions = self.track.possible_actions(St)

                self.pi[y, x, y_vel, x_vel][a_max] = 1 - \
                    self.eps + self.eps/len(possible_actions)

                a_not_max = [a for a in possible_actions if a != a_max]
                self.pi[y, x, y_vel, x_vel][tuple(
                    zip(*a_not_max))] = self.eps/len(possible_actions)

            it += 1


rt = RaceTrack.from_csv("../racetracks/map1.csv")

agent = OnPolicyMonteCarloAgent(rt, n_episodes=3)
S, A, R = agent.generate_episode(agent.pi)
print(S)
