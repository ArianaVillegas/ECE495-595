import base64
import imageio
import IPython
import matplotlib.pyplot as plt


def plot_return(num_iterations, eval_interval, global_returns, labels, filename):
    iterations = range(0, num_iterations + 1, eval_interval)

    print(global_returns)
    plt.clf()

    for label in labels:
        plt.plot(iterations, global_returns[label], label=label) 

    plt.legend()
    plt.grid()
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)
    plt.savefig(filename)


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


def create_policy_eval_video(env, algo, filename, random=False, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = env.reset()
            video.append_data(env.render())
            while not time_step.is_last():
                action_step = algo.get_action([time_step.observation], random)
                time_step = env.step(action_step.action)
                video.append_data(env.render())
    return embed_mp4(filename)