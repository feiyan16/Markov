import random


class State:
    def __init__(self, state_id, reward):
        self.id = state_id
        self.reward = reward
        self.actions: dict = {}
        self.optimal_a = -1

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


print_format = "(s{} a{} {:.4f}) "
states = read_file("/Users/feiyansu/Desktop/test2.in")
iteration = []
print_str = "After iteration 1: "
# base cases, at t = 1
for state in states:
    iteration.insert(state.id, state.reward)
    rand_a = random.randint(1, len(state.actions))
    print_str += (print_format.format(state.id, rand_a, state.reward))
print(print_str)
# remaining cases from 2-20
for t in range(1, 20):
    print_str = "After iteration " + str(t+1) + ": "
    iteration_t = []
    for state in states:
        max_a = -1000
        opt_a = None
        for action, state_prob in state.actions.items():
            x_val = 0
            for sp in state_prob:
                prob = sp[1]
                j = iteration[sp[0]-1]
                x_val += (prob * j)
            max_a = max(max_a, x_val)
            if max_a == x_val:
                opt_a = action
        iteration_t.insert(state.id, state.reward + 0.9 * max_a)
        print_str += (print_format.format(state.id, opt_a, iteration_t[state.id - 1]))
    iteration = iteration_t.copy()
    print(print_str)