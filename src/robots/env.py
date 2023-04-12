from typing import Union, cast
import gym
import torch
import numpy as np
from gym import spaces
import robosuite

from robosuite.environments import MujocoEnv


class RobotEnv:

    _env: MujocoEnv

    robot: str = 'Panda'
    # robot: str = 'Sawyer'

    # task: str = 'Lift'
    task: str = 'Door'

    x_size: int
    a_size: int

    observation_space: spaces.Box
    action_space: spaces.Box

    def __init__(self, *args, **kwargs) -> None:

        self._env = robosuite.make(
            env_name=self.task,
            robots=self.robot,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            *args, **kwargs
        )

        self.x_size = 0
        for val in self._env.observation_spec().values():
            if isinstance(val, np.ndarray):
                self.x_size += sum(val.shape)
            elif isinstance(val, int) or isinstance(val, float):
                self.x_size += 1

        # self.x_size = sum(sum(p.shape) for p in self._env.observation_spec().values())
        boundary = np.array([np.inf] * self.x_size)
        self.observation_space = spaces.Box(low=-boundary, high=boundary)

        low, high = self._env.action_spec
        self.a_size = low.shape[0]
        self.action_space = spaces.Box(low=low, high=high)

    def _get_observation(self, obs: dict):

        arrays = []
        for val in obs.values():
            if isinstance(val, np.ndarray):
                arrays += [val]
            elif isinstance(val, int) or isinstance(val, float):
                arrays += [np.array([val])]
        array = np.concatenate(arrays)

        return array

    def reset(self):
        obs = self._env.reset()
        obs = self._get_observation(obs)
        return obs

    def step(self, action: Union[np.ndarray, torch.Tensor]):
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        obs, reward, done, info = self._env.step(action)
        obs = self._get_observation(obs)
        return obs, reward, done, info

    def render(self):
        return self._env.render()
