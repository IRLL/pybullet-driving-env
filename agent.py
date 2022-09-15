from typing import *
import torch
from dataclasses import dataclass
from torch.distributions.normal import Normal
from config import Config
import numpy as np


class PolicyNetwork(torch.nn.Module):
    """Gaussian policy network"""

    def __init__(
        self, num_input: int, num_output: int, num_hidden: int
    ) -> None:
        super().__init__()
        self.input = torch.nn.Linear(num_input, num_hidden)
        self.fully_connected = torch.nn.Linear(num_hidden, num_hidden)
        self.mean = torch.nn.Linear(num_hidden, num_output)
        self.log_std = torch.nn.Parameter(torch.zeros(1, num_output))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.distributions.normal.Normal, torch.Tensor]:
        # Input layer
        x = self.input(x)
        x = torch.tanh(x)

        # Hidden layer
        x = self.fully_connected(x)
        x = torch.tanh(x)

        # Output layer
        mean = self.mean(x)
        std = self.log_std.exp()
        dist = torch.distributions.normal.Normal(mean, std)

        return dist, mean


class ValueNetwork(torch.nn.Module):
    """Value network that quantifies the quality of an action given a state."""

    def __init__(self, num_input: int, num_hidden: int):
        super().__init__()
        self.input = torch.nn.Linear(num_input, num_hidden)
        self.fully_connected = torch.nn.Linear(num_hidden, num_hidden)
        self.output = torch.nn.Linear(num_hidden, 1)

    def forward(self, x: torch.Tensor):
        # Input layer
        x = self.input(x)
        x = torch.tanh(x)

        # Hidden layer
        x = self.fully_connected(x)
        x = torch.tanh(x)

        # Output layer
        value = self.output(x)

        return value


@dataclass
class NormalizationParams:
    """Normalization paramters for state and rewards"""

    mean_state: torch.Tensor
    var_state: torch.Tensor
    mean_reward: torch.Tensor
    var_reward: torch.Tensor


def initialize_weight(param) -> None:
    """Initialization of the weight's value of a network"""
    if isinstance(param, torch.nn.Linear):
        torch.nn.init.orthogonal_(param.weight.data)
        torch.nn.init.constant_(param.bias.data, 0)


class Normalization:
    """Normalize the states and rewards on the fly"""

    def __init__(
        self,
        dtype: torch.FloatTensor = torch.float,
        epsilon: float = 1e-4,
        nums: int = 1,
    ):
        self.mean = torch.zeros(1, nums, dtype=dtype)
        self.var = torch.ones(1, nums, dtype=dtype)
        self.count = epsilon

    def update(self, x: torch.Tensor):
        batch_mean = torch.mean(x, axis=0)
        batch_var = torch.var(x, axis=0)
        batch_count = x.size(dim=0)
        self.update_mean_var(batch_mean, batch_var, batch_count)

    def update_mean_var(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int,
    ):
        self.mean, self.var, self.count = self.update_mean_var_count(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @staticmethod
    def update_mean_var_count(
        mean: torch.Tensor,
        var: torch.Tensor,
        count: int,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int,
    ) -> Tuple[torch.Tensor, int]:
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        ma = var * count
        mb = batch_var * batch_count
        m2 = ma + mb + (delta**0.5) * count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


class Agent:
    """PPO agent"""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        config: Config,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config

        self.policy_network = PolicyNetwork(
            num_input=self.num_states,
            num_hidden=self.config.num_hiddens,
            num_output=self.num_actions,
        )

        self.value_network = ValueNetwork(
            num_input=num_states,
            num_hidden=self.config.num_hiddens,
        )

        # # Intialize networks's parameters
        self.policy_network.apply(initialize_weight)
        self.value_network.apply(initialize_weight)

        # Get optimizer for two models
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.config.learning_rate,
        )

        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=self.config.learning_rate,
        )

        self.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.policy_optimizer, gamma=self.config.decay_coef
        )
        self.value_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.value_optimizer, gamma=self.config.decay_coef
        )

    def select_action(self, state: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        with torch.no_grad():
            dist, mu = self.policy_network(state)

        return dist, mu

    def compute_value(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            value = self.value_network(state)

        return value.detach().cpu().numpy()

    def optimize(
        self, policy_loss: torch.Tensor, value_loss: torch.Tensor
    ) -> None:
        # Policy net
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.policy_optimizer.step()

        # Value net
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
        self.value_optimizer.step()
