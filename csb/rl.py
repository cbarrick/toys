import time


def rl_loop(env, agent, n=1000000, learn=True, render=0):
    '''A loop for reinforcement learning.

    Args:
        env (gym.Env):
            The environment of the experiment. See the package `csb.envs` for
            preprocessors and other useful wrappers for environments.
        agent:
            The agent to train. An agent provides two methods: `act` and
            `learn`. The act method takes in an observation and returns an
            action. The learn method updates the model from the observation,
            the action taken, the resulting observation, the immediate reward,
            a boolean indicating whether the episode is done, and an info
            dictionary for debugging.
        n (int):
            The number of episodes to run the loop.
        learn (bool):
            Set to False to skip the learning step.
        render (int):
            The frequency at which to render the environment. A value of 0
            turns off rendering (the default).
    '''
    frame_count = 0

    for ep_i in range(n):
        start_frame = frame_count
        start_time = time.perf_counter()
        total_reward = 0

        obs = env.reset()
        done = False

        while not done:
            if render and frame_count % render == 0:
                env.render()

            action = agent.act(obs)
            obs_next, reward, done, info = env.step(action)
            if learn:
                agent.learn(obs, action, obs_next, reward, done, info)

            obs = obs_next
            total_reward += reward
            frame_count += 1

        frames = frame_count - start_frame
        elapsed = time.perf_counter() - start_time
        print('episode:', ep_i, end='\t')
        print('reward:', total_reward, end='\t')
        print('frames:', frames, end='\t')
        print('obs/s:', '{:.2f}'.format(frames / elapsed), end='\t')
        print()

    if render:
        env.render(close=True)
