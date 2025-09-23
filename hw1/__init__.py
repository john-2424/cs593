"""CS 593 RL1 HW1 package.
Submodules:
  networks      - Neural network architectures (MLP only)
  datasets      - Vector dataset utilities
  collector     - Demonstration collection
  imitation     - Behavioral cloning learner
"""
from .networks import MLPNetwork
from .datasets import VectorDataset
from .collector import Collector
from .imitation import ImitationLearner

__all__ = [
    'MLPNetwork','VectorDataset','Collector','ImitationLearner'
]
