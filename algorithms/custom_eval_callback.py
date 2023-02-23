import gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, eval_freq: int, n_eval_episodes: int, verbose=0):
        super(CustomEvalCallback, self).__init__(verbose)

        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.learning_curve = list()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        mean_reward, std_reward = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=True
        )
        self.learning_curve.append((mean_reward, std_reward))
        return True
