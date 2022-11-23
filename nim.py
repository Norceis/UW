import statistics
import itertools

class Nim():

    def __init__(self, n_rows: int = 4):
        self._initial_state = [int((x + 1)) for x in range(0, n_rows * 2, 2)]
        self._possible_values_in_rows = []
        for idx in self._initial_state:
            temp_list = []
            for ldx in range(idx+1):
                temp_list.append(ldx)
            self._possible_values_in_rows.append(temp_list)

        self._states = (set(itertools.product(*self._possible_values_in_rows)))
        self._n_states = len(self._states)
        self._current_state = self._initial_state
        self._transition_probs = dict()
        # print(len(self._states))
        # self._states.remove((0,0,0,0))
        # print(len(self._states))

    def fill_transition_probs(self):
        for state in self._states:
            self._transition_probs[state] = dict()
            actions = self.get_possible_actions(state)
            for action in actions:
                self._transition_probs[state][action] = dict()
                new_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(state, action)])
                self._transition_probs[state][action][new_state] = 1

    def reset(self):
        self._current_state = self._initial_state
        return self._current_state

    def get_all_states(self):
        return self._states

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
        """ return a set of possible next states and probabilities of moving into them """
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)
        return self._transition_probs[state][action]

    def get_number_of_states(self):
        return self._n_states

    def get_reward(self, state, action, next_state):
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)

        reward = -1
        if self.is_terminal(next_state):
            reward = 100

        return reward

    def step(self, action):
        prev_state = self._current_state
        self._current_state = [idx_1 - idx_2 for idx_1, idx_2 in zip(self._current_state, action)]
        return self._current_state, self.get_reward(prev_state, action, self._current_state), \
               self.is_terminal(self._current_state), None



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

nim = Nim(4)
optimal_policy, optimal_value = value_iteration(nim, 0.9, 0.001)
# print(optimal_policy, optimal_value)