import gymnasium as gym
import numpy as np
from collections import deque
import cv2

class FrameStackResize(gym.Wrapper):
    """
    Wrapper for frame stacking and resizing image observations.
    Stacks multiple consecutive frames and resizes them to reduce computational overhead.
    """
    
    def __init__(self, env, num_stack=4, resize_shape=(84, 84)):
        """
        Initialize the frame stack wrapper.
        
        Args:
            env: The environment to wrap
            num_stack: Number of frames to stack
            resize_shape: Shape to resize frames to (height, width)
        """
        super(FrameStackResize, self).__init__(env)
        self.num_stack = num_stack
        self.resize_shape = resize_shape
        
        # Create new observation space for stacked and resized frames
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(self.num_stack * 3, *self.resize_shape),  # 3 channels (RGB) × num_stack
            dtype=np.uint8
        )
        
        self.frames = deque([], maxlen=self.num_stack)
    
    def reset(self, **kwargs):
        """Reset the environment and stack initial frames."""
        observation, info = self.env.reset(**kwargs)
        
        # Process the first frame
        processed_frame = self._process_frame(observation)
        
        # Fill the frame buffer with the initial frame
        for _ in range(self.num_stack):
            self.frames.append(processed_frame)
            
        # Stack frames
        stacked_frames = self._get_stacked_frames()
        
        return stacked_frames, info
    
    def step(self, action):
        """Take a step and stack the resulting frame."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Process the new frame
        processed_frame = self._process_frame(observation)
        
        # Add to frame buffer
        self.frames.append(processed_frame)
        
        # Stack frames
        stacked_frames = self._get_stacked_frames()
        
        return stacked_frames, reward, terminated, truncated, info
    
    def _process_frame(self, frame):
        """
        Process a single frame:
        1. Resize the frame
        2. Convert to CHW format (PyTorch format)
        """
        # Resize frame
        resized = cv2.resize(frame, self.resize_shape, interpolation=cv2.INTER_AREA)
        
        # Convert from HWC to CHW format
        processed = np.transpose(resized, (2, 0, 1))
        
        return processed
    
    def _get_stacked_frames(self):
        """Stack frames into a single observation."""
        return np.concatenate(list(self.frames), axis=0)