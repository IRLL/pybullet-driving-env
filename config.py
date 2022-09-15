from dataclasses import dataclass

from numpy import gradient


@dataclass
class Config:
    """Hyperparameters for neural networks"""

    learning_rate: float = 3e-4
    batch_size: int = 64
    num_env_steps: int = 2048
    num_epochs: int = 10
    num_iter: int = 1000
    gamma: float = 0.99
    num_hiddens: int = 64
    lambda_gae: float = 0.95
    clipping_coef: float = 0.1
    value_loss_coef: float = 0.5
    decay_coef: float = 0.99
    grad_norm: float = 0.5
    device: str = "cpu"
