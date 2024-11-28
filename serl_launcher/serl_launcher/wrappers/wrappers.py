import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
from collections import deque
import cv2
import imageio
import os

class PixelFrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, stack_key='pixels'):
        super().__init__(env)
        self._num_frames = num_frames
        self.stack_key = stack_key
        self._frames = deque([], maxlen=num_frames)
        pixels_shape = env.observation_space[stack_key].shape
        

        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self.observation_space[stack_key] = Box(low=0, high=255, shape=(num_frames*pixels_shape[-1], *pixels_shape[:-1]), dtype=np.uint8)

    def _transform_observation(self, obs):
        assert len(self._frames) == self._num_frames
        obs[self.stack_key] = np.concatenate(list(self._frames), axis=0)
        return obs

    def _extract_pixels(self, obs):
        pixels = obs[self.stack_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        obs, info = self.env.reset()
        pixels = self._extract_pixels(obs)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels = self._extract_pixels(obs)
        self._frames.append(pixels)
        return self._transform_observation(obs), reward, terminated, truncated, info
    
class StateFrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, stack_key='state', flatten=True):
        super().__init__(env)
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)   
        self.stack_key = stack_key
        self.flatten = flatten

        shape = self.env.observation_space[stack_key].shape
        if isinstance(shape, int):
            shape = (shape,)  # Convert to a tuple for consistency
        else:
            shape = shape  # If it's already a tuple or list, keep it as is
        if flatten: 
            self.observation_space[stack_key] = Box(low=-np.inf, high=np.inf, shape=(num_frames * shape[-1],), dtype=np.float32)
        else:
            self.observation_space[stack_key] = Box(low=-np.inf, high=np.inf, shape=(num_frames, *shape), dtype=np.float32)

    def _transform_observation(self):
        assert len(self._frames) == self._num_frames
        obs = np.array(self._frames)
        if self.flatten:
            obs = obs.flatten()
        return obs

    def reset(self):
        obs, info = self.env.reset()
        for _ in range(self._num_frames):
            self._frames.append(obs[self.stack_key])
        obs[self.stack_key] = self._transform_observation()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs[self.stack_key])
        obs[self.stack_key] = self._transform_observation()
        return obs, reward, terminated, truncated, info

class CustomPixelObservation(gym.ObservationWrapper):
  """Resize the observation to a given resolution"""
  def __init__(self, env, crop_resolution=None, resize_resolution=None):
    super().__init__(env)
    if isinstance(resize_resolution, int):
      resize_resolution = (resize_resolution, resize_resolution)
    if isinstance(crop_resolution, int):
      crop_resolution = (crop_resolution, crop_resolution)
    self.crop_resolution = crop_resolution
    self.resize_resolution = resize_resolution
    self.observation_space['pixels'] = Box(low=0, high=255, shape=(*self.resize_resolution, 3), dtype=np.uint8)
    
  def observation(self, observation):
    if self.crop_resolution is not None:
      if observation['pixels'].shape[:2] != self.crop_resolution:
        center = observation['pixels'].shape
        x = center[1]/2 - self.crop_resolution[1]/2
        y = center[0]/2 - self.crop_resolution[0]/2
        observation['pixels']= observation['pixels'][int(y):int(y+self.crop_resolution[0]), int(x):int(x+self.crop_resolution[1])]
    if self.resize_resolution is not None:
      if observation['pixels'].shape[:2] != self.resize_resolution:
        observation['pixels'] = cv2.resize(
            observation['pixels'],
            dsize=self.resize_resolution,
            interpolation=cv2.INTER_CUBIC,
        )
    return observation    

class VideoRecorder(gym.Wrapper):
  """Wrapper for rendering and saving rollouts to disk.
  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(
      self,
      env,
      save_dir,
      crop_resolution,
      resize_resolution,
      fps = 20,
      current_episode=0,
      record_every=2,
  ):
    super().__init__(env)

    self.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    num_vids = len(os.listdir(save_dir))
    current_episode = num_vids*record_every

    if isinstance(resize_resolution, int):
      self.resize_resolution = (resize_resolution, resize_resolution)
    if isinstance(crop_resolution, int):
      self.crop_resolution = (crop_resolution, crop_resolution)

    self.resize_h, self.resize_w = self.resize_resolution
    self.crop_h, self.crop_w = self.crop_resolution
    self.fps = fps
    self.enabled = True
    self.current_episode = current_episode
    self.record_every = record_every
    self.frames = []

  def step(self, action):
    observation, reward, terminated, truncated, info = self.env.step(action)
    if self.current_episode % self.record_every == 0:
      frame = self.env.render()[0]
      if self.crop_resolution is not None:
        # Crop
        if frame.shape[:2] != (self.crop_h, self.crop_w):
          center = frame.shape
          x = center[1]/2 - self.crop_w/2
          y = center[0]/2 - self.crop_h/2
          frame = frame[int(y):int(y+self.crop_h), int(x):int(x+self.crop_w)]
      if self.resize_resolution is not None:
        if frame.shape[:2] != (self.resize_h, self.resize_w):
          frame = cv2.resize(
              frame,
              dsize=(self.resize_h, self.resize_w),
              interpolation=cv2.INTER_CUBIC,
          )
      # Write rewards on the frame
      cv2.putText(
          frame,
          f"{reward:.3f}",
          (10, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          1,
          cv2.LINE_AA,
      )
      # Save
      self.frames.append(frame)
    if terminated or truncated:
      if self.current_episode % self.record_every == 0:
        filename = os.path.join(self.save_dir, f"{self.current_episode}.mp4")
        imageio.mimsave(filename, self.frames, fps=self.fps)
        self.frames = []
      self.current_episode += 1
    return observation, reward, terminated, truncated, info
  
class ActionRepeat(gym.Wrapper):
  """Repeat the agent's action N times in the environment.
  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env, repeat):
    """Constructor.
    Args:
      env: A gym env.
      repeat: The number of times to repeat the action per single underlying env
        step.
    """
    super().__init__(env)

    assert repeat > 1, "repeat should be greater than 1."
    self._repeat = repeat

  def step(self, action):
    total_reward = 0.0
    for _ in range(self._repeat):
      observation, reward, terminated, truncated, info = self.env.step(action)
      total_reward += reward
      if terminated or truncated:
        break
    return observation, total_reward, terminated, truncated, info
  
class FrankaObservation(gym.ObservationWrapper):
  """Resize the observation to a given resolution"""
  def __init__(self, env, camera_name='front'):
    super().__init__(env)
    self.camera_name = camera_name
    pixel_space = self.observation_space['images'][camera_name]
    self.state_keys = ['panda/tcp_pos', 'panda/tcp_orientation', 'panda/gripper_pos', 'panda/gripper_vec']
    state_dim = 0
    for key in self.state_keys:
      state_dim += self.observation_space['state'][key].shape[0]
    state_space = Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)
    self.observation_space = Dict({'pixels': pixel_space, 'state': state_space})
    
  def observation(self, observation):
    pixels = observation['images'][self.camera_name]
    state = np.concatenate([observation['state'][key] for key in self.state_keys])
    observation = {}
    observation['pixels'] = pixels
    observation['state'] = state
    return observation    
  
class ActionState(gym.Wrapper):
    # Add previous action to the state
    def __init__(self, env, state_key='state', action_key='action'):
        super().__init__(env)
        self.action_key = action_key
        self.state_key = state_key
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space[state_key].shape[0]
        self.observation_space[state_key] = Box(low=-np.inf, high=np.inf, shape=(self.state_dim + self.action_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        self.action = np.zeros(self.action_dim)
        obs[self.state_key] = np.concatenate([obs[self.state_key], self.action])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs[self.state_key] = np.concatenate([obs[self.state_key], action])
        return obs, reward, terminated, truncated, info