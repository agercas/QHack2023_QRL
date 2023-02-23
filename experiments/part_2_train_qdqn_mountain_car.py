"""
For original paper and code see https://github.com/qdevpsi3/qrl-dqn-gym
"""

import argparse
import pickle

import gym
import torch
import torch.nn as nn
import yaml
from torch.nn.parameter import Parameter

from qrldqngym.common.qnn import get_model
from qrldqngym.common.trainer import Trainer

parser = argparse.ArgumentParser()

parser.add_argument("--n_layers", default=5, type=int)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--w_input", default=True, type=bool)
parser.add_argument("--w_output", default=True, type=bool)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--lr_input", default=0.001, type=float)
parser.add_argument("--lr_output", default=0.1, type=float)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--eps_init", default=1.0, type=float)
parser.add_argument("--eps_decay", default=0.99, type=int)
parser.add_argument("--eps_min", default=0.01, type=float)
parser.add_argument("--train_freq", default=10, type=int)
parser.add_argument("--target_freq", default=30, type=int)
parser.add_argument("--memory", default=10000, type=int)
parser.add_argument("--data_reupload", default=True, type=bool)
parser.add_argument("--loss", default="SmoothL1", type=str)
parser.add_argument("--optimizer", default="RMSprop", type=str)
parser.add_argument("--total_episodes", default=5000, type=int)
parser.add_argument("--n_eval_episodes", default=5, type=int)
parser.add_argument("--logging", default=True, type=bool)
parser.add_argument("--log_train_freq", default=1, type=int)
parser.add_argument("--log_eval_freq", default=20, type=int)
parser.add_argument("--log_ckp_freq", default=50, type=int)
parser.add_argument("--device", default="auto", type=str)
args = parser.parse_args()


class QuantumNet(nn.Module):
    def __init__(self, n_layers, w_input, w_output, data_reupload):
        super(QuantumNet, self).__init__()
        self.n_qubits = 2
        self.n_actions = 3
        self.data_reupload = data_reupload
        self.q_layers = get_model(n_qubits=self.n_qubits, n_layers=n_layers, data_reupload=data_reupload)
        # convert from 2 qubits to 3 actions
        # not adding more complexity here because we want to learn through quantum circuit
        self.layer1 = nn.Linear(2, 3)

        if w_input:
            self.w_input = Parameter(torch.Tensor(self.n_qubits))
            nn.init.normal_(self.w_input)
        else:
            self.register_parameter("w_input", None)
        if w_output:
            self.w_output = Parameter(torch.Tensor(self.n_actions))
            nn.init.normal_(self.w_output, mean=90.0)
        else:
            self.register_parameter("w_output", None)

    def forward(self, inputs):
        if self.w_input is not None:
            inputs = inputs * self.w_input
        inputs = torch.atan(inputs)
        q_outputs = self.q_layers(inputs)
        q_outputs = (1 + q_outputs) / 2

        outputs = self.layer1(q_outputs)

        if self.w_output is not None:
            outputs = outputs * self.w_output
        else:
            outputs = 90 * outputs
        return outputs


def main():
    # Environment
    env_name = "MountainCar-v0"
    env = gym.make(env_name)

    # Networks
    net = QuantumNet(args.n_layers, args.w_input, args.w_output, args.data_reupload)
    target_net = QuantumNet(args.n_layers, args.w_input, args.w_output, args.data_reupload)

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
        learning_rate_input=args.lr_input,
        learning_rate_output=args.lr_output,
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

    with open("trainer_mountain_car.pkl", "wb") as outp:
        pickle.dump(trainer, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
