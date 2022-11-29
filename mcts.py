import statistics
import itertools
import numpy as np
import random
import copy
from collections import defaultdict

class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self.win_states = list()
        for idx in range(len(self.state)):
            list_of_zeros = [0] * len(self.state)
            list_of_zeros[idx] = 1
            self.win_states.append(tuple(list_of_zeros))
        self.win_states = tuple(self.win_states)
        print('chujc')

    def untried_actions(self):
        self._untried_actions = list(self.get_legal_actions(self.state))
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):

        action = self._untried_actions.pop()
        next_state = self.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.is_game_over(self.state)

    def rollout(self):
        current_rollout_state = self.state

        while not self.is_game_over(current_rollout_state):

            possible_moves = self.get_legal_actions(current_rollout_state)

            action = self.rollout_policy(possible_moves)
            current_rollout_state = self.move(action)
        return self.game_result(current_rollout_state)

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):

        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):

        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 100

        for i in range(simulation_no):

            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        return self.best_child(c_param=0.)

    def get_legal_actions(self, state):
        possible_actions = []

        if self.is_game_over(state):
            possible_actions.append(self.state)
            return tuple(possible_actions)

        for row_idx, number_in_row in enumerate(state):
            for number in range(1, number_in_row + 1):
                single_action = [0 for _ in range(len(self.state))]
                single_action[row_idx] = number
                single_action = tuple(single_action)
                possible_actions.append(single_action)

        return tuple(possible_actions)

    def is_game_over(self, state):
        if not any(state): return True
        if sum(state) == 1: return True
        return False

    def game_result(self, state):
        if self.is_game_over(state):
            return -1

        elif state in self.win_states:
            return 1

        else:
            return 0

    def move(self, action):
        return tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(self.state, action)])

root = MonteCarloTreeSearchNode(state=(1,3,5,7))
root.untried_actions()
selected_node = root.best_action()
selected_node