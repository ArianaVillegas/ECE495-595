import pyvirtualdisplay

import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.specs import tensor_spec

from replay_buffer import ReplayBuffer
from utils import plot_return, create_policy_eval_video
from train import train
from dqn import DQN, DDQN

# Set up a virtual display for rendering OpenAI gym environments.
display = pyvirtualdisplay.Display(visible=1, size=(1400, 900)).start()

tf.version.VERSION

num_iterations = 10000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1 # @param {type:"integer"}
replay_buffer_max_length = 10000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

env_name = 'CartPole-v0'
env = suite_gym.load(env_name)
env.reset()

fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

global_returns = {}
optimizers = [
    tf.keras.optimizers.Adam(learning_rate=learning_rate),
    tf.keras.optimizers.SGD(learning_rate=learning_rate*10)
]
algos = [DQN, DDQN]

with open('results_final.txt', 'a') as f:
    for ALGO in algos:
        local_returns = {}
        f.write(f'========= {ALGO} =========\n')
        print(f'========= {ALGO} =========')
        for optimizer in optimizers:
            algo = ALGO(fc_layer_params, num_actions, optimizer)
            algo_name = algo.name()
            name = optimizer._name
            f.write(f'-------- {name} --------\n')
            print(f'-------- {name} --------')
            buffer = ReplayBuffer(replay_buffer_max_length)
            buffer.fill_buffer(env, algo, initial_collect_steps)
            returns = train(env, 
                            algo, 
                            buffer, 
                            num_iterations=num_iterations, 
                            batch_size=batch_size, 
                            log_interval=log_interval, 
                            num_eval_episodes=num_eval_episodes, 
                            eval_interval=eval_interval)
            f.write(','.join(list(map(str, returns))))
            print(','.join(list(map(str, returns))))
            f.write('\n')
            local_returns[name] = returns
            del algo
            del buffer
        global_returns[algo_name] = local_returns

plot_return(num_iterations, eval_interval, global_returns['DDQN'], ['SGD'], 'ddqn_sgd_2.png')

plot_return(num_iterations, eval_interval, global_returns['DQN'], ['Adam'], '1a.png')
plot_return(num_iterations, eval_interval, global_returns['DQN'], global_returns['DQN'].keys(), '1b.png')
plot_return(num_iterations, eval_interval, {
                'DQN': global_returns['DQN']['Adam'],
                'DDQN': global_returns['DDQN']['Adam']
            }, 
            ['DQN', 'DDQN'], '2a.png')
plot_return(num_iterations, eval_interval, global_returns['DDQN'], global_returns['DDQN'].keys(), '2b.png')

# create_policy_eval_video(env, algo, 'trained-agent', random=False)
# create_policy_eval_video(env, algo, 'random-agent', random=True)