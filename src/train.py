from replay_buffer import ReplayBuffer
from dqn import DQN

def compute_avg_return(env, algo, num_episodes=10):
  total_return = 0.0
  for _ in range(num_episodes):
    time_step = env.reset()
    state = time_step.observation
    episode_return = 0.0

    while not time_step.is_last():
        action = algo.get_action([state])
        time_step = env.step(action)
        state = time_step.observation
        reward = time_step.reward
        episode_return += reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return


# Config dict with the training setup
# - num_iterations
# - batch_size
# - log_interval
# - num_eval_episodes
# - eval_interval
def train(env, algo, buffer, **kwargs):
    avg_return = compute_avg_return(env, algo, kwargs['num_eval_episodes'])
    returns = [avg_return]
    for step in range(kwargs['num_iterations']):
        buffer.record_experience(env, algo)
        experience_batch = buffer.get_batch_experience(kwargs['batch_size'])
        train_loss = algo.train(experience_batch)

        if step % kwargs['log_interval'] == 0:
            algo.update_target_net()
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % kwargs['eval_interval'] == 0:
            avg_return = compute_avg_return(env, algo, kwargs['num_eval_episodes'])
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    return returns