from typing import *
import torch
from agent import Agent
from config import Config
import numpy as np


class PPO:
    """Proximal policy optimization"""

    def __init__(
        self, env, num_states: int, num_actions: int, config: Config
    ) -> None:
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config
        self.agent: Agent = Agent(
            num_states=num_states, num_actions=num_actions, config=config
        )
        self._device = torch.device(self.config.device)
        self._dtype = torch.float32

    def update_parameters(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advs: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> Tuple[float, float]:
        """Update policy & value networks"""

        returns = advs + values
        num_states = len(returns)
        for _ in range(self.config.num_epochs):
            for _ in range(num_states // self.config.batch_size):
                # TODO: Send data to device (need to test with cuda)
                idx = np.random.randint(0, num_states, self.config.batch_size)
                (
                    state,
                    action,
                    return_,
                    adv,
                    old_value,
                    old_log_prob,
                ) = self.get_batch(
                    states=states,
                    actions=actions,
                    returns=returns,
                    advs=advs,
                    values=values,
                    log_probs=log_probs,
                    idx=idx,
                )
                state = state.to(self._device)
                action = action.to(self._device)
                return_ = return_.to(self._device)
                adv = adv.to(self._device)
                old_value = old_value.to(self._device)
                old_log_prob = old_log_prob.to(self._device)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Compute the value and values loss
                value = self.agent.value_network(state)
                value_loss = (
                    torch.nn.functional.mse_loss(return_, value)
                    * self.config.value_loss_coef
                )

                # Get action distribution & the log-likelihood
                action_dist, _ = self.agent.policy_network(state)
                new_log_prob = action_dist.log_prob(action)
                ratio = torch.exp(new_log_prob - old_log_prob)

                # Compute policy loss
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * torch.clamp(
                    ratio,
                    1 - self.config.clipping_coef,
                    1 + self.config.clipping_coef,
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                self.agent.optimize(
                    policy_loss=policy_loss, value_loss=value_loss
                )

        return policy_loss.item(), value_loss.item()

    def train(self) -> None:
        # state = self.env.reset()
        eps = 0
        cummul_rew = 0
        eps_rew = 0
        eps_rews = []
        tot_num_steps = 0
        step = 0
        max_action = self.env.action_space.high
        min_action = self.env.action_space.low

        for i in range(self.config.num_iter):
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []
            step = 0

            spawn_position = np.random.uniform(-10, 10, (3,))
            spawn_orientation = np.random.uniform(-1, 1, (4,))
            spawn_position[2] = 0.5
            obs = self.env.reset(
                goal=12 * np.ones((2,)),
                base_position=spawn_position,
                base_orientation=spawn_orientation,
            )
            car = obs["car_qpos"]
            gridmap = obs["segmentation"]
            state = np.concatenate(
                (obs["car_qpos"].flatten(), obs["segmentation"].flatten()),
                axis=0,
            )
            # state = self.env.reset(seed=0)

            for _ in range(self.config.num_env_steps):
                # Select an action
                dist, _ = self.agent.select_action(
                    torch.tensor(state, dtype=torch.float32).view(1, -1)
                )
                action = dist.sample().cpu().numpy()[0]

                # Send action to env
                clipped_action = np.clip(action, min_action, max_action)
                obs, reward, done, _ = self.env.step(clipped_action)
                next_state = np.concatenate(
                    (obs["car_qpos"].flatten(), obs["segmentation"].flatten()),
                    axis=0,
                )
                cummul_rew += reward

                # Collect data
                states.append(torch.tensor(state, dtype=torch.float32))
                actions.append(torch.tensor(action, dtype=torch.float32))
                rewards.append(torch.tensor(reward, dtype=torch.float32))
                dones.append(torch.tensor(done, dtype=torch.float32))

                if done:
                    obs = self.env.reset(
                        goal=12 * np.ones((2,)),
                        base_position=spawn_position,
                        base_orientation=spawn_orientation,
                    )
                    state = np.concatenate(
                        (
                            obs["car_qpos"].flatten(),
                            obs["segmentation"].flatten(),
                        ),
                        axis=0,
                    )
                    # state = self.env.reset()
                    eps = eps + 1
                    eps_rews.append(cummul_rew)
                    cummul_rew = 0
                    step = 0
                else:
                    state = next_state
                    step += 1

            # Convert to torch
            states = torch.vstack(states)
            actions = torch.vstack(actions)
            rewards = torch.vstack(rewards)
            dones = torch.vstack(dones)

            # Make dataloader
            batch_state = self.make_dataloader(
                states, self.config.batch_size, self.num_states
            )
            batch_action = self.make_dataloader(
                actions, self.config.batch_size, self.num_actions
            )
            # Compute values
            values = self.compute_value(batch_state, dones)

            # Compute the values for the last state
            with torch.no_grad():
                next_value = self.agent.value_network(
                    torch.tensor(next_state, dtype=torch.float32).view(1, -1)
                )
            next_value = next_value * (1 - done)
            values = torch.cat((values, next_value), dim=0)

            # Compute log-likelihood
            log_probs = self.compute_log_lik(batch_state, batch_action)

            # Compute the generalized advantage estimation
            advs = self.compute_gae(rewards=rewards, values=values, dones=dones)

            # Update parameters
            policy_loss, value_loss = self.update_parameters(
                states=states,
                actions=actions,
                advs=advs,
                values=values[:-1],
                log_probs=log_probs,
            )

            if i % 1 == 0:
                avg_rewards = np.mean(
                    eps_rews[np.maximum(len(eps_rews) - 100, 0) :]
                )
                print(
                    f"iter #{i}/{self.config.num_iter}, policy loss: {policy_loss:0.2f}, value loss: {value_loss:0.2f}, avg reward: {avg_rewards: .2f}"
                )

    def make_dataloader(
        self, dataset: torch.Tensor, batch_size: int, num_obs: int
    ) -> List[torch.Tensor]:
        """Create a dataloader in batches"""

        # Initialization
        output_batches = torch.zeros((batch_size, num_obs), dtype=self._dtype)
        num_data = len(dataset)
        data_loader = []
        count = 0
        for i, y in enumerate(dataset):
            output_batches[count, :] = y

            # Store data
            if (i + 1) % batch_size == 0:
                data_loader.append(output_batches)

                # Reset
                count = 0
                output_batches = torch.zeros(
                    (batch_size, num_obs), dtype=self._dtype
                )
            else:
                count += 1
                if i == num_data - 1:
                    data_loader.append(output_batches[:count, :])

        return data_loader

    def compute_value(
        self, observations: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute values given the states"""
        values = []
        for obs in observations:
            with torch.no_grad():
                values.append(self.agent.value_network(obs))

        return torch.vstack(values)

    def compute_log_lik(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute the log likelihood for each actions"""
        log_probs = []
        for obs, action in zip(observations, actions):
            with torch.no_grad():
                dist, _ = self.agent.policy_network(obs)
            log_probs.append(dist.log_prob(action))
        return torch.vstack(log_probs)

    @staticmethod
    def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma=0.99,
        lam=0.95,
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation. See equations 11 & 12 in
        https://arxiv.org/pdf/1707.06347.pdf"""

        advs = []
        gae = 0.0
        dones = torch.cat((dones, torch.zeros(1, 1)), dim=0)
        for s in reversed(range(len(rewards))):
            delta = (
                rewards[s]
                + gamma * (values[s + 1]) * (1 - dones[s])
                - values[s]
            )
            gae = delta + gamma * lam * (1 - dones[s]) * gae
            advs.append(gae)
        advs.reverse()
        return torch.vstack(advs)

    @staticmethod
    def get_batch(
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advs: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        idx: np.ndarray,
    ) -> Tuple[torch.Tensor]:
        """Randomly get a batch of data"""

        return (
            states[idx],
            actions[idx],
            returns[idx],
            advs[idx],
            values[idx],
            log_probs[idx],
        )
