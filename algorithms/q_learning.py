from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from algorithms.custom_eval_callback import CustomEvalCallback


class QLearning(BaseAlgorithm):
    """
    Q-Learning
    This implementation follows the Hugging Face DRL course: https://huggingface.co/deep-rl-course/
    NOTE: this implementation has some unused parameters because we inherit from the SB3 BaseAlgorithm class.
    """

    def _setup_model(self) -> None:
        pass

    def __init__(
        self,
        policy: str,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        gamma: float = 0.95,  # Discounting rate
        max_steps: int = 99,  # Max steps per episode
        max_epsilon: float = 1.0,  # Exploration probability at start
        min_epsilon: float = 0.05,  # Minimum exploration probability
        decay_rate: float = 0.0005,  # Exponential decay rate for exploration prob
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            verbose=verbose,
            device=device,
            seed=seed,
        )

        # Agent parameters
        self.gamma = gamma
        self.max_steps = max_steps

        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

        state_space_n = self.observation_space.n
        action_space_n = self.action_space.n
        self.q_table = np.zeros((state_space_n, action_space_n))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "QL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        step = 0
        episode_n = 0
        while step < total_timesteps:
            episode_n += 1
            episode_step = 0
            episode_done = False
            # Reduce epsilon (because we need less and less exploration)
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode_n)
            state = self.env.reset()

            # Run an episode
            while not episode_done or episode_step < self.max_steps:
                step += 1
                episode_step += 1

                action = self._epsilon_greedy_policy(state, epsilon)

                # Take action At and observe Rt+1 and St+1
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, episode_done, info = self.env.step(action)
                self._update_q_table(state, action, new_state, reward)

                # Our next state is the new state
                state = new_state

                if isinstance(callback, CustomEvalCallback):
                    if step % callback.eval_freq == 0:
                        callback.on_step()

    def _greedy_policy(self, state):
        # Exploitation: take the action with the highest state, action value
        action = np.array([np.argmax(self.q_table[s][:]) for s in state])
        return action

    def _epsilon_greedy_policy(self, state, epsilon):
        # Randomly generate a number between 0 and 1
        random_int = np.random.uniform(0, 1)
        # if random_int > greater than epsilon --> exploitation
        if random_int > epsilon:
            action = self._greedy_policy(state)
        # else --> exploration
        else:
            action = [self.action_space.sample() for _ in state]
        return action

    def _update_q_table(self, state, action, new_state, reward):
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        for s, a, n, r in zip(state, action, new_state, reward):
            self.q_table[s][a] = self.q_table[s][a] + self.learning_rate * (
                r + self.gamma * np.max(self.q_table[n]) - self.q_table[s][a]
            )

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if deterministic:
            action = self._greedy_policy(observation)
            return action, state

        else:
            raise NotImplementedError
