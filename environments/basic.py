import gym
import gym.spaces
import numpy as np
import scipy as sp
import scipy.ndimage


class BoxNormalize(gym.ObservationWrapper):
    '''A preprocessor to normalize box observations.
    '''

    def __init__(self, env, bounds=(-1, 1)):
        super().__init__(env)
        obsp = self.observation_space
        self.shift = bounds[0]
        self.scale = (bounds[1] - bounds[0]) / (obsp.high - obsp.low)
        self.low = obsp.low
        self.observation_space = gym.spaces.Box(bounds[0], bounds[1], obsp.shape)

    def _observation(self, obs):
        return self.shift + self.scale * (obs - self.low)


class Grayscale(gym.ObservationWrapper):
    '''A preprocessor to convert RGB observations to grayscale.
    '''

    def __init__(self, env):
        super().__init__(env)
        obsp = self.observation_space
        low = self._observation(obsp.low)
        high = self._observation(obsp.high)
        self.observation_space = gym.spaces.Box(low, high)

    def _observation(self, obs):
        r = obs[..., 0]
        g = obs[..., 1]
        b = obs[..., 2]
        return ((0.2126 * r) + (0.7152 * g) + (0.0722 * b))


class Deflicker(gym.ObservationWrapper):
    '''A preprocessor that maxes each frame with the previous frame.
    '''

    def __init__(self, env):
        super().__init__(env)
        self.last_frame = np.zeros(self.observation_space.shape)

    def _observation(self, obs):
        maxed = np.max(obs, self.last_frame)
        self.last_frame = obs
        return obs

    def _reset(self):
        self.last_frame.fill(0)
        return self.env.reset()


class Zoom(gym.ObservationWrapper):
    '''A preprocessor for resizing image observations.
    '''

    def __init__(self, env, zoom, **kwargs):
        super().__init__(env)
        self.zoom = zoom
        self.kwargs = kwargs
        obsp = self.observation_space
        low = self._observation(obsp.low)
        high = self._observation(obsp.high)
        self.observation_space = gym.spaces.Box(low, high)

    def _observation(self, obs):
        return sp.ndimage.zoom(obs, self.zoom, **self.kwargs)


class BoxActionRepeat(gym.Wrapper):
    '''A wrapper that repeates the action for some frames.

    The resulting observation is the stack of frames.
    '''

    def __init__(self, env, repeat):
        super().__init__(env)
        self.repeat = repeat
        rr = self.reward_range
        self.reward_range = (rr[0] * repeat, rr[1] * repeat)
        obsp = self.observation_space
        new_shape = (*obsp.shape, repeat)
        low = np.broadcast_to(np.expand_dims(obsp.low, -1), new_shape)
        high = np.broadcast_to(np.expand_dims(obsp.high, -1), new_shape)
        self.observation_space = gym.spaces.Box(low, high)

    def _step(self, action):
        total_reward = 0
        obs = np.zeros(self.observation_space.shape)
        for i in range(self.repeat):
            frame, reward, done, info = self.env.step(action)
            total_reward += reward
            obs[..., i] = frame
            if done: break
        return obs, total_reward, done, info

    def _reset(self):
        blank = np.zeros(self.observation_space.shape)
        frame = np.expand_dims(self.env.reset(), axis=-1)
        return blank + frame
