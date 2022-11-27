import statistics
import itertools

class Nim():

    def __init__(self, n_rows: int = 4):
        self.initial_state = [int((x + 1)) for x in range(0, n_rows * 2, 2)]
        self.possible_values_in_rows = []
        for idx in self.initial_state:
            temp_list = []
            for ldx in range(idx+1):
                temp_list.append(ldx)
            self.possible_values_in_rows.append(temp_list)

        self.states = (set(itertools.product(*self.possible_values_in_rows)))
        self.n_states = len(self.states)
        self.current_state = self.initial_state
        self.transition_probs = dict()

    def fill_transition_probs(self):
        for state in self.states:
            self.transition_probs[state] = dict()
            actions = self.get_possible_actions(state)
            for action in actions:
                self.transition_probs[state][action] = dict()
                new_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(state, action)])
                self.transition_probs[state][action][new_state] = 1

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
        return self.transition_probs[state][action]

    def get_number_of_states(self):
        return self.n_states

    def get_reward(self, state, action, next_state):
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)

        reward = -1
        if self.is_terminal(next_state):
            reward = 100

        return reward

    def step(self, action):
        prev_state = self.current_state
        self.current_state = [idx_1 - idx_2 for idx_1, idx_2 in zip(self.current_state, action)]
        return self.current_state, self.get_reward(prev_state, action, self.current_state), \
               self.is_terminal(self.current_state), None



def value_iteration(nim, gamma, theta):
    V = dict()
    nim.fill_transition_probs()

    for state in nim.get_all_states():
        V[state] = 0

    policy = dict()
    for current_state in nim.get_all_states():
        try:
            policy[current_state] = nim.get_possible_actions(current_state)[0]
        except IndexError:
            continue

    while True:
        last_mean_value = statistics.fmean(V.values())
        for current_state in nim.get_all_states():
            actions = nim.get_possible_actions(current_state)
            state_action_values = dict()

            for action in actions:

                state_action_values[action] = 0
                next_states = nim.get_next_states(current_state, action)

                for next_state in next_states:
                    state_action_values[action] += next_states[next_state] * (nim.get_reward(current_state, action, next_state) + gamma * V[next_state])

            V[current_state] = max(list(state_action_values.values()))
            print(abs(statistics.fmean(V.values()) - last_mean_value), theta)
        if abs(statistics.fmean(V.values()) - last_mean_value) < theta:
            break

    for current_state in nim.get_all_states():

        state_action_values = dict()
        actions = nim.get_possible_actions(current_state)

        for action in actions:


            state_action_values[action] = 0
            next_states = nim.get_next_states(current_state, action)


            for next_state in next_states:
                state_action_values[action] += next_states[next_state] * (nim.get_reward(current_state, action, next_state) + gamma * V[next_state])

        max_value_action = max(state_action_values, key=state_action_values.get)

        if policy[current_state] != max_value_action:
            policy[current_state] = max_value_action

    return policy, V

# nim = Nim(4)
# optimal_policy, optimal_value = value_iteration(nim, 0.9, 0.001)
# print(optimal_policy, optimal_value)

import random
from collections import defaultdict


class ExpectedSARSAAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    #---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        #
        # INSERT CODE HERE to get maximum possible value for a given state
        #

        return max([self.get_qvalue(state, action) for action in possible_actions])

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * \sum_a \pi(a|s') Q(s', a))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        sum_of_strategies = list()

        [sum_of_strategies.append(1 / len(self.get_legal_actions(next_state)) * self.get_qvalue(next_state, action)) for action in self.get_legal_actions(next_state)]

        value = (1 - learning_rate) * self.get_qvalue(state, action) + learning_rate * (reward + gamma * sum(sum_of_strategies))
        self.set_qvalue(state, action, value)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        possible_actions_dict = dict()

        for action in possible_actions:
            possible_actions_dict[action] = self.get_qvalue(state, action)

        sorted_dict = sorted(possible_actions_dict.items(), key=lambda kv: kv[1])

        return random.choice([k for k, v in possible_actions_dict.items() if v == sorted_dict[-1][-1]])

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        if random.random() < epsilon:
            return random.choice(possible_actions)

        return self.get_best_action(state)

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0

def play_and_train(env, agent):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    state = env.reset()

    done = False

    while not done:
        # get agent to pick action given state state.
        action = agent.get_action(state)

        next_state, reward, done, _ = env.step(action)

        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        if done:
            break

    return total_reward

# nim = Nim()
# agent = ExpectedSARSAAgent(alpha=0.1, epsilon=0.1, discount=0.99,
#                        get_legal_actions=nim.get_possible_actions)
#
# for i in range(10000):
#     play_and_train(nim, agent)
#
# agent.turn_off_learning()
#
# for i in range(10):
#     print(play_and_train(nim, agent))

# play at random
nim = Nim()
player_1_wins = 0
player_2_wins = 0
games = 0

for _ in range(1000):
    while not nim.is_terminal(nim.current_state):
        random_action = random.choice(nim.get_possible_actions(nim.current_state))
        nim.step(random_action)
        if nim.is_terminal(nim.current_state):
            games += 1
            if not games % 2:
                player_2_wins += 1
            player_1_wins += 1

print(player_1_wins, player_2_wins, games)