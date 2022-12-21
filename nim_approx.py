import statistics
import itertools
import numpy as np
import random
import copy
from collections import defaultdict

from matplotlib import pyplot as plt


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):

        self.get_legal_actions = get_legal_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.weights = [0.5, 0.5]

    def function_bits(self, state):

        first_part = -1 if not int(nim_sum(state)) else 1
        second_part = 1 if state in nim.win_states else -1

        return first_part, second_part

    def get_qvalue(self, state, action):

        first_part, second_part = self.function_bits(state)

        return first_part * self.weights[0] + second_part * self.weights[1]

    def get_value(self, state):

        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return 0.0

        return max([self.get_qvalue(state, action) for action in possible_actions])

    def update(self, state, action, reward, next_state):

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        possible_actions = nim.get_possible_actions(state)
        next_states = [tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(state, action)]) for action in possible_actions]
        next_states_qvalues = [self.get_qvalue(next_state, action) for next_state in next_states]
        # print(next_states_qvalues)

        flags = self.function_bits(state)
        error = (reward + gamma * max(next_states_qvalues) - self.get_qvalue(state, action))

        for idx in range(len(self.weights)):
            self.weights[idx] += learning_rate * error * flags[idx]

        # for idx in range(len(self.weights)):
        #     if self.weights[idx] > 1:
        #         self.weights[idx] = 1
        #     elif self.weights[idx] < 0:
        #         self.weights[idx] = 0

    def get_best_action(self, state):

        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return None

        possible_actions_dict = dict()

        for action in possible_actions:
            possible_actions_dict[action] = self.get_qvalue(state, action)

        sorted_dict = sorted(possible_actions_dict.items(), key=lambda kv: kv[1])

        return random.choice([k for k, v in possible_actions_dict.items() if v == sorted_dict[-1][-1]])

    def get_action(self, state):

        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return None

        epsilon = self.epsilon

        if random.random() < epsilon:
            return random.choice(possible_actions)

        return self.get_best_action(state)

    def turn_off_learning(self):
        self.epsilon = 0
        self.alpha = 0


class Nim:

    def __init__(self, n_rows: int = 4):
        self.initial_state = tuple([int((x + 1)) for x in range(0, n_rows * 2, 2)])
        self.possible_values_in_rows = []
        for idx in self.initial_state:
            temp_list = []
            for ldx in range(idx + 1):
                temp_list.append(ldx)
            self.possible_values_in_rows.append(temp_list)

        self.states = (tuple(itertools.product(*self.possible_values_in_rows)))
        self.n_states = len(self.states)
        self.current_state = copy.deepcopy(self.initial_state)
        self.win_states = list()
        for idx in range(n_rows):
            list_of_zeros = [0] * n_rows
            list_of_zeros[idx] = 1
            self.win_states.append(tuple(list_of_zeros))
        self.win_states = tuple(self.win_states)

    def reset(self):
        self.current_state = self.initial_state
        return self.current_state

    def get_all_states(self):
        return self.states

    def is_terminal(self, state):
        if not any(state): return True
        return False

    def get_possible_actions(self, state):
        possible_actions = []

        if self.is_terminal(state):
            possible_actions.append(state)
            return tuple(possible_actions)

        for row_idx, number_in_row in enumerate(state):
            for number in range(1, number_in_row + 1):
                single_action = [0 for _ in range(len(state))]
                single_action[row_idx] = number
                single_action = tuple(single_action)
                possible_actions.append(single_action)

        return tuple(possible_actions)

    def get_next_states(self, state, action):
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)
        next_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(state, action)])
        return next_state

    def get_number_of_states(self):
        return self.n_states

    def get_reward(self, state, action, next_state):
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)

        reward = 0

        if self.is_terminal(next_state):
            reward = -0.05

        if next_state in self.win_states:
            reward = 0.05

        if not int(nim_sum(next_state)):
            reward = 0.05

        return reward

    def step(self, action):
        prev_state = self.current_state
        self.current_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(self.current_state, action)])
        return self.current_state, self.get_reward(prev_state, action, self.current_state), \
               self.is_terminal(self.current_state), None


def nim_sum(state):
    binary_rows = [format(row, 'b') for row in state]
    max_len = max([len(row) for row in binary_rows])
    binary_rows = ['0' * (max_len - len(row)) + row for row in binary_rows]

    result = ''
    for idx in range(max_len):
        res = 0
        for row in binary_rows:
            res += int(row[idx])
            res %= 2
        result += str(res)
    return result


def play_and_train_ql(env, agent, player=0):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    state = env.reset()

    done = False
    turn = 0 if not player else 1

    while not done:
        if not turn % 2:
            # get agent to pick action given state state.
            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward
        else:
            action = random.choice(env.get_possible_actions(state))
            next_state, reward, done, _ = env.step(action)
            state = next_state

        if done:
            turn += 1
            break

    return total_reward


nim = Nim()

agent_ql_first = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                                get_legal_actions=nim.get_possible_actions)

agent_ql_second = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                                 get_legal_actions=nim.get_possible_actions)
rewards = []
for i in range(10000):
    print(agent_ql_first.weights)
    rewards.append(play_and_train_ql(nim, agent_ql_first, 0))
    play_and_train_ql(nim, agent_ql_second, 1)

# play ql vs random
player_1_wins = 0
player_2_wins = 0

for _ in range(10000):
    nim.reset()
    turn = 0
    while not nim.is_terminal(nim.current_state):
        if not turn % 2:
            action = agent_ql_first.get_best_action(nim.current_state)
        else:
            action = random.choice(nim.get_possible_actions(nim.current_state))

        nim.step(action)

        if nim.is_terminal(nim.current_state):
            if turn % 2:
                player_1_wins += 1
            else:
                player_2_wins += 1

        turn += 1

print(f'Algorithm winrate: {player_1_wins * 100 / (player_1_wins + player_2_wins)}%')
plt.plot(rewards, linewidth=0.05)
plt.ylabel('some numbers')
plt.show()