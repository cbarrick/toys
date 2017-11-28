import gym

from .basic import BoxActionRepeat
from .basic import BoxNormalize
from .basic import Deflicker
from .basic import Grayscale
from .basic import Zoom


class Atari(gym.Wrapper):
    '''A wrapper for Atari environments to apply preprocessing:
    - Down sampling: Frames are converted to grayscale and resized.
    - Action repeat: Actions are repeated for some number of frames.
    - Frame stacking: Observations are the stack of frames skipped over.
    - Deflicker: Frames are maxed with the previous frame. This is useful in
      games where sprites are flickered between frames (off by default).
    '''

    def __init__(self, env, zoom=0.5, repeat=4, deflicker=False):
        env = gym.make(env) if isinstance(env, str) else env
        env = BoxNormalize(env)
        env = Grayscale(env)
        env = Deflicker(env) if deflicker else env
        env = Zoom(env, zoom)
        env = BoxActionRepeat(env, repeat)
        super().__init__(env)
