import random
import sys


class State:
    def __init__(self, state_id, reward):
        self.id = state_id
        self.reward = reward
        self.actions: dict = {}
        self.optimal_a = -1
        self.j_t = -1

    def add_action(self, action_id, action_s, action_prob):
        if action_id in self.actions.keys():
            updated = self.actions.get(action_id)
            updated.append((action_s, action_prob))
        else:
            self.actions[action_id] = [(action_s, action_prob)]


def read_file(filename):
    states_ = []  # stores all parse lines as states
    # open file and read in lines
    file = open(filename, "r")
    file_data = file.readlines()
    for data in file_data:
        # check if line is empty
        if data.isspace() or len(data) == 0:
            continue
        id_reward = ""
        actions = ""
        for idx in range(0, len(data)):
            if data[idx] == '(':
                id_reward = data[0:idx]
                actions = data[idx:]
                break
        # check if split happened
        if len(id_reward) == 0 or len(actions) == 0:
            continue
        id_reward = id_reward.split(" ")
        # check if tokens have id and reward
        if len(id_reward) < 2:
            continue
        # check if id is empty
        if id_reward[0].isspace() or len(id_reward[0]) == 0:
            continue
        # check if reward is empty
        if id_reward[1].isspace() or len(id_reward[1]) == 0:
            continue
        new_state = State(int(id_reward[0][1:]), float(id_reward[1]))
        actions = actions.split(")")
        for a in actions:
            a = a.strip()
            a = a[1:]
            a = a.split(" ")
            # check if a is in format (action state probability)
            if len(a) < 3:
                continue
            action_id = a[0][1:]
            action_s = a[1][1:]
            action_prob = a[2]
            new_state.add_action(int(action_id), int(action_s), float(action_prob))
        states_.append(new_state)
    return states_


def printer(updated_states, it_t):
    print_str = ""
    print("After iteration " + str(it_t) + ": ")
    for s in updated_states:
        print_str += ("(s{} a{} {:.4f}) ".format(s.id, s.optimal_a, s.j_t))
    print(print_str)


def bellman_calculator(g, states):
    iteration = [None] * max(state.id for state in states)
    # base cases, at t = 1
    for state in states:
        # add Ji = ri to row at timestamp = 1
        iteration[state.id - 1] = state.reward
        # select random action
        rand_a = random.randint(1, len(state.actions))
        # set optimal action and j value
        state.optimal_a = rand_a
        state.j_t = state.reward
    # print iteration
    printer(states, 1)

    # remaining cases from 2-20
    for t in range(1, 20):
        # j values at timestamp = t
        iteration_t = iteration.copy()
        for state in states:
            max_a = -1000
            # dict to map max_a to relevant action
            max_to_a = {}
            # for (action: [(state, probability) ... ]) in {}
            for action, state_prob in state.actions.items():
                x_val = 0  # E(x)
                # for (state, probability) in []
                for sp in state_prob:
                    prob = sp[1]  # probability to get to state
                    j = iteration[sp[0] - 1]  # J(state) at t-1
                    try:
                        x_val += (prob * j)  # calculate E(x)
                    except TypeError:
                        print("s" + str(sp[0]) + " does not exists, action cannot reach it.")
                # calculate max(a)
                max_a = max(max_a, x_val)
                # if max_a has been updated
                if max_a not in max_to_a.keys():
                    max_to_a[max_a] = action
            # set optimal_a to final max_a
            state.optimal_a = max_to_a[max_a]
            # reset J(state)
            state.j_t = state.reward + g * max_a
            # add j values to row t
            iteration_t[state.id - 1] = state.j_t
        # update "dp table" with new row t
        iteration = iteration_t.copy()
        # print iteration t
        printer(states, t + 1)


def main():
    n = len(sys.argv)
    if n < 5:
        print("python3 main.py <number of states> <max actions> <input file path> <discount factor>")
        quit()
    elif n == 5:
        gamma = sys.argv[4]
        file_path = sys.argv[3]
        all_states = read_file(file_path)
        bellman_calculator(float(gamma), all_states)


if __name__ == "__main__":
    main()
