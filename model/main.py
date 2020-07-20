from typing import List, Tuple
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
import webbrowser
from trainer import Trainer
import subprocess

@dataclass
class ParseOptions:
    M1_dim: Tuple[int, int] = field(metadata=dict(args=["-M1_dim"]), default=(2,2))
    M2_dim: Tuple[int, int] = field(metadata=dict(args=["-M2_dim"]), default=(2, 2))
    hidden_layers: str = field(metadata=dict(args=["-hiddens"]), default=None)
    log_dir: str = field(metadata=dict(args=["-log-dir"]), default=None)
    learning_rate: float = field(metadata=dict(args=["-learning_rate"]), default=1e-3)
    buffer_size: int = field(metadata=dict(args=["-buf_size"]), default=1000)
    batch_size: int = field(metadata=dict(args=["-batch_size"]), default=32)
    loss: str = field(metadata=dict(args=["-loss"]), default="mse")
    optimizer: str = field(metadata=dict(args=["-optimizer"]), default="adam")
    activation: str = field(metadata=dict(args=["-activation"]), default="ReLU")
    layer: str = field(metadata=dict(args=["-layer"]), default="affine")


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    hiddens = args.hidden_layers[1:len(args.hidden_layers) - 1]
    hiddens = hiddens.split(',')
    hiddens = [int(i) for i in hiddens]
    args.hidden_layers = hiddens
    subprocess.Popen(["tensorboard",  "--logdir", args.log_dir])
    webbrowser.open("127.0.0.1:6006")
    trainer = Trainer(**vars(args))
    trainer.train()
