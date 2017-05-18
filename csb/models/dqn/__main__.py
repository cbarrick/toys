import argparse

from csb.envs import Atari
from csb.rl import rl_loop

from . import dqn

# yapf: disable
parser = argparse.ArgumentParser(description='Run the DQN model.')
parser.add_argument('-n', type=int, default=1000000, help='The number of episodes')
parser.add_argument('-e', '--env', type=str, default='Breakout-v0', help='The environment to train/demo')
parser.add_argument('-r', '--render', metavar='N', type=int, default=0, help='Render every N observations')
parser.add_argument('-d', '--demo', action='store_true', help='Only demo, do not train')

args = parser.parse_args()
env = Atari(args.env)
obs_shape = env.observation_space.shape
n_actions = env.action_space.n
agent = dqn.DQN(obs_shape, n_actions)

rl_loop(env, agent, n=args.n, learn=not args.demo, render=args.render)
