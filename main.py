import random
import sys


class State:
    def __init__(self, state_id, reward):
        self.id = state_id
        self.reward = reward
        self.actions: dict = {}
        self.optimal_a = -1
        self.j = -1

    def add_action(self, action_id, action_s, action_prob):
        if action_id in self.actions.keys():
            updated = self.actions.get(action_id)
            updated.append((action_s, action_prob))
        else:
            self.actions[action_id] = [(action_s, action_prob)]


def read_file(filename):
    states_ = []
    file = open(filename, "r")
    file_data = file.readlines()
    for data in file_data:
        id_reward = ""
        actions = ""
        for idx in range(0, len(data)):
            if data[idx] == '(':
                id_reward = data[0:idx]
                actions = data[idx:]
                break
        new_state = State(int(id_reward.split(" ")[0][1:]), float(id_reward.split(" ")[1]))
        actions = actions.split(")")
        for a in actions:
            a = a.strip()
            a = a[1:]
            a = a.split(" ")
            if len(a) == 3:
                action_id = a[0][1:]
                action_s = a[1][1:]
                action_prob = a[2]
                new_state.add_action(int(action_id), int(action_s), float(action_prob))
        states_.append(new_state)
    return states_


def printer(updated_states, it_t):
    print_str = "After iteration " + str(it_t) + ": "
    for s in updated_states:
        print_str += ("(s{} a{} {:.4f}) ".format(s.id, s.optimal_a, s.j))
    print(print_str)


def bellman_calculator(g, states):
    iteration = []
    # base cases, at t = 1
    for state in states:
        iteration.insert(state.id, state.reward)
        rand_a = random.randint(1, len(state.actions))
        state.optimal_a = rand_a
        state.j = state.reward
    printer(states, 1)
    # remaining cases from 2-20
    for t in range(1, 20):
        iteration_t = []
        for state in states:
            max_a = -1000
            for action, state_prob in state.actions.items():
                x_val = 0
                for sp in state_prob:
                    prob = sp[1]
                    j = iteration[sp[0] - 1]
                    x_val += (prob * j)
                max_a = max(max_a, x_val)
                if max_a == x_val:
                    state.optimal_a = action
            state.j = state.reward + g * max_a
            iteration_t.insert(state.id, state.j)
        printer(states, t+1)
        iteration = iteration_t.copy()


n = len(sys.argv)
if n < 5:
    print("python3 main.py <number of states> <max actions> <input file path> <discount factor>")
    quit()
elif n == 5:
    gamma = sys.argv[4]
    file_path = sys.argv[3]
    all_states = read_file(file_path)
    bellman_calculator(float(gamma), all_states)
