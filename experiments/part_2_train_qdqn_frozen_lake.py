"""
For original paper and code see https://github.com/qdevpsi3/qrl-dqn-gym
"""
import argparse
import pickle

import gym
import numpy as np
import torch.nn as nn
import yaml

from qrldqngym.common.qnn import get_model
from qrldqngym.common.trainer import Trainer
from qrldqngym.common.wrappers import BinaryWrapper

parser = argparse.ArgumentParser()

parser.add_argument("--n_layers", default=5, type=int)
parser.add_argument("--gamma", default=0.8, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--batch_size", default=11, type=int)
parser.add_argument("--eps_init", default=1.0, type=float)
parser.add_argument("--eps_decay", default=0.99, type=int)
parser.add_argument("--eps_min", default=0.01, type=float)
parser.add_argument("--train_freq", default=5, type=int)
parser.add_argument("--target_freq", default=10, type=int)
parser.add_argument("--memory", default=10000, type=int)
parser.add_argument("--loss", default="SmoothL1", type=str)
parser.add_argument("--optimizer", default="RMSprop", type=str)
parser.add_argument("--total_episodes", default=10000, type=int)
parser.add_argument("--n_eval_episodes", default=5, type=int)
parser.add_argument("--logging", default=True, type=bool)
parser.add_argument("--log_train_freq", default=1, type=int)
parser.add_argument("--log_eval_freq", default=20, type=int)
parser.add_argument("--log_ckp_freq", default=50, type=int)
parser.add_argument("--device", default="auto", type=str)
args = parser.parse_args()


class QuantumNet(nn.Module):
    def __init__(self, n_layers):
        super(QuantumNet, self).__init__()
        self.n_qubits = 4
        self.n_actions = 4
        self.q_layers = get_model(n_qubits=self.n_qubits, n_layers=n_layers, data_reupload=False)

    def forward(self, inputs):
        inputs = inputs * np.pi
        outputs = self.q_layers(inputs)
        outputs = (1 + outputs) / 2
        return outputs


def main():
    # Environment
    env_name = "FrozenLake-v1"
    env = gym.make(env_name)
    env = BinaryWrapper(env)

    # Networks
    net = QuantumNet(args.n_layers)
    target_net = QuantumNet(args.n_layers)

    # Trainer
    trainer = Trainer(
        env,
        net,
        target_net,
        gamma=args.gamma,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        exploration_initial_eps=args.eps_init,
        exploration_decay=args.eps_decay,
        exploration_final_eps=args.eps_min,
        train_freq=args.train_freq,
        target_update_interval=args.target_freq,
        buffer_size=args.memory,
        loss_func=args.loss,
        optim_class=args.optimizer,
        device=args.device,
        logging=args.logging,
    )

    if args.logging:
        with open(trainer.log_dir + "config.yaml", "w") as f:
            yaml.safe_dump(args.__dict__, f, indent=2)

    trainer.learn(
        args.total_episodes,
        n_eval_episodes=args.n_eval_episodes,
        log_train_freq=args.log_train_freq,
        log_eval_freq=args.log_eval_freq,
        log_ckp_freq=args.log_ckp_freq,
    )

    with open("trainer_frozen_lake.pkl", "wb") as outp:
        pickle.dump(trainer, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
