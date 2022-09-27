import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

WORLD_SIZE = 5 #square
GAMMA = 0.9 #discount rate
ACTION_PROB = 0.25 #pi(action, state)

A_POS = [4,4] #[0, 1] # max reward pos
A_PRIME_POS = [4, 1]
B_POS = [0,0]  #[0, 3] # max reward pos
B_PRIME_POS = [2, 3]

# N, S, E, W
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTIONS_FIGS = ['←', '↑', '→', '↓']

# def step(state, action):
#     if state == A_POS:
#         return A_PRIME_POS, 10
#     if state == B_POS:
#         return B_PRIME_POS, 5

#     next_state = (np.array(state) + action).tolist()
#     x, y = next_state
#     #if x and y out of world size give negative reward
#     if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
#         reward = -1.0
#         next_state = state
#     else:
#         reward = 0
#     return next_state, reward


# def draw_image(image):
#     fig, ax = plt.subplots()
#     ax.set_axis_off()
#     tb = Table(ax, bbox=[0, 0, 1, 1])

#     nrows, ncols = image.shape
#     width, height = 1.0 / ncols, 1.0 / nrows

#     # Add cells
#     for (i, j), val in np.ndenumerate(image):
#         tb.add_cell(i, j, width, height, text=val,
#                     loc='center', facecolor='white')

#     # Row and column labels...
#     for i in range(len(image)):
#         tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
#                     edgecolor='none', facecolor='none')
#         tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
#                     edgecolor='none', facecolor='none')

#     ax.add_table(tb)


# def draw_policy(optimal_values):
#     fig, ax = plt.subplots()
#     ax.set_axis_off()
#     tb = Table(ax, bbox=[0, 0, 1, 1])

#     nrows, ncols = optimal_values.shape
#     width, height = 1.0 / ncols, 1.0 / nrows

#     # Add cells
#     for (i, j), val in np.ndenumerate(optimal_values):
#         next_vals = []
#         for action in ACTIONS:
#             next_state, _ = step([i, j], action)
#             next_vals.append(optimal_values[next_state[0], next_state[1]])

#         best_actions = np.where(next_vals == np.max(next_vals))[0]
#         val = ''
#         for ba in best_actions:
#             val += ACTIONS_FIGS[ba]
#         tb.add_cell(i, j, width, height, text=val,
#                     loc='center', facecolor='white')

#     # Row and column labels...
#     for i in range(len(optimal_values)):
#         tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
#                     edgecolor='none', facecolor='none')
#         tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
#                     edgecolor='none', facecolor='none')

#     ax.add_table(tb)

# def gridworld_iterations():
#     value = np.zeros((WORLD_SIZE, WORLD_SIZE))
#     while True:
#         # keep iteration until convergence
#         new_value = np.zeros_like(value)
#         for i in range(WORLD_SIZE):
#             for j in range(WORLD_SIZE):
#                 for action in ACTIONS:
#                     (next_i, next_j), reward = step([i, j], action)
#                     # bellman equation
#                     new_value[i, j] += ACTION_PROB * (reward + GAMMA * value[next_i, next_j])
#         if np.sum(np.abs(value - new_value)) < 1e-4:
#             draw_image(np.round(new_value, decimals=2))
#             plt.savefig('images/figure_3_2.png')
#             plt.show()
#             plt.close()
#             break
#         value = new_value

# def gridworld_policy():
#     value = np.zeros((WORLD_SIZE, WORLD_SIZE))
#     while True:
#         # keep iteration until convergence
#         new_value = np.zeros_like(value)
#         for i in range(WORLD_SIZE):
#             for j in range(WORLD_SIZE):
#                 values = []
#                 for action in ACTIONS:
#                     (next_i, next_j), reward = step([i, j], action)
#                     # value iteration
#                     values.append(reward + GAMMA * value[next_i, next_j])
#                 new_value[i, j] = np.max(values)
#         if np.sum(np.abs(new_value - value)) < 1e-4:
#             draw_image(np.round(new_value, decimals=2))
#             plt.savefig('images/figure_3_5.png')
#             plt.show()
#             plt.close()
#             draw_policy(new_value)
#             plt.savefig('images/figure_3_5_policy.png')
#             plt.show()
#             plt.close()
#             break
#         value = new_value

# if __name__ == '__main__':
#     gridworld_iterations()
#     gridworld_policy()
