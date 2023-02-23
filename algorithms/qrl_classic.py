from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from algorithms.custom_eval_callback import CustomEvalCallback


class QRLClassic(BaseAlgorithm):
    """
    Quantum Inspired TD-lambda modified algorithm
    Implemented without quantum circuits. Algorithm details https://arxiv.org/abs/0810.3828
    NOTE: we do not recommend using this algorithm, see the part 1 results section for details
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
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_steps = max_steps

        state_space_n = self.observation_space.n
        action_space_n = self.action_space.n

        # value function V (same as in TD-lambda)
        self.v = np.zeros(state_space_n)
        # this is the f(s) = |a> function, an equivalent of a q-table
        self.f_table = np.ones((state_space_n, action_space_n)) / np.sqrt(action_space_n)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "QRL",
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
        while step < total_timesteps:
            episode_step = 0
            episode_done = False
            state = self.env.reset()

            # Run an episode
            while not episode_done or episode_step < self.max_steps:
                step += 1
                episode_step += 1

                action = self._observe_action(state)
                new_state, reward, episode_done, info = self.env.step(action)

                # NOTE: we change the condition for L from what it was in the paper!
                # The new condition is simpler and for the specific problem is sufficient
                # L = min(int(k * (r + v[new_state])), int(np.pi / theta))
                delta = []
                for s, a, n, r in zip(state, action, new_state, reward):
                    delta.append(r + self.gamma * self.v[n] - self.v[s])
                L = [1 if d > 0 else 0 for d in delta]

                self._update_value_function(state, new_state, reward)
                self._update_probability_amplitudes(state, action, L)

                # Our next state is the new state
                state = new_state

                if isinstance(callback, CustomEvalCallback):
                    if step % callback.eval_freq == 0:
                        callback.on_step()

    def _observe_action(self, state):
        observed = []
        for s in state:
            # quantum measurement on f
            actions = self.f_table[s]
            probabilities = np.array([a * a for a in actions])
            action = np.random.choice(range(len(probabilities)), p=probabilities)
            observed.append(action)
        return observed

    def _update_value_function(self, state, new_state, reward):
        # V(s) := V(s) + lr [r + gamma V(s') - V(s')]
        for s, n, r in zip(state, new_state, reward):
            self.v[s] = self.v[s] + self.learning_rate * (r + self.gamma * self.v[n] - self.v[s])

    def _update_probability_amplitudes(self, state, action, L):
        n = self.action_space.n
        for s, a, l in zip(state, action, L):
            U_a = np.identity(n)
            U_a[a][a] = -1

            acs = np.array([self.f_table[s]])
            U = 2 * np.matmul(np.transpose(acs), acs) - np.identity(n)

            U_grover = np.matmul(U, U_a)

            na = self.f_table[s]
            for i in range(l):
                na = np.matmul(U_grover, na)

            self.f_table[s] = na

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if deterministic:
            action = self._observe_action(observation)
            return action, state

        else:
            raise NotImplementedError
