"""
Training package for fluid dynamics models.
"""

from .trainer import FluidTrainer
from .config import TrainingConfig
from .callbacks import FluidTrainingCallbacks
from .utils import (
    setup_training,
    run_training,
    evaluate_model
)

__all__ = [
    'FluidTrainer',
    'TrainingConfig', 
    'FluidTrainingCallbacks',
    'setup_training',
    'run_training',
    'evaluate_model'
]
