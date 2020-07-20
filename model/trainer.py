import torch
torch.manual_seed(42)
import torch.nn as nn
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from model import MLP, ProdMLP
from typing import List, Tuple
from collections import deque
from tqdm import trange
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

class Trainer:
    def __init__(self, M1_dim: Tuple[int, int], M2_dim: Tuple[int, int],
        hidden_layers: List[int], log_dir: str, learning_rate: float = 1e-3, batch_size: int = 32,
        buffer_size: int =1000, n_steps: int = int(1e6), val_every = 1e4, loss: str = "mse", optimizer: str = "adam", activation: str = "ReLU", layer = "affine", x_min = -100, x_max = 100, **kwargs):

        self.M1_dim = M1_dim
        self.M2_dim = M2_dim
        self.x_min = x_min
        self.x_max = x_max

        self.out_dim = (M1_dim[0], M2_dim[1])
        if layer.lower() in ("prod", "product"):
            self.mlp = ProdMLP(M1_dim = M1_dim, M2_dim = M2_dim, hiddens = hidden_layers, activation = activation)
        else:
            self.mlp = MLP(M1_dim = M1_dim, M2_dim = M2_dim, hiddens = hidden_layers, activation = activation)

        self.lr = learning_rate
        if optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.mlp.parameters(),
						   lr=self.lr,
						   betas=(0.9, 0.999),
						   eps=1e-08,
						   weight_decay=0,
						   amsgrad=False)
        else:
            print("using SGD instead of Adam")
            self.optimizer = optim.SGD(MLP.parameters, lr = self.lr)
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "huber":
            self.loss = nn.SmoothL1Loss()
        else:
            print(f"{loss} not supported, using MSE")
            self.loss = nn.MSELoss()
        self.val_loss = nn.L1Loss()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)

        self.n_steps = n_steps
        self.val_every = val_every

        self.writer = SummaryWriter(log_dir)

    # fills the buffer with randomly created examples from which to train on
    def _fill_buffer(self):
        for _ in range(self.buffer_size):
            self.buffer.append(self._make_data())

    def _make_data(self):
        M1 = torch.rand(self.M1_dim) * random.randint(self.x_min, self.x_max)
        M2 = torch.rand(self.M2_dim) * random.randint(self.x_min, self.x_max)
        x = torch.cat((M1.reshape(-1), M2.reshape(-1)), dim=0)
        y = (M1 @ M2).reshape(-1)
        return x, y

    def _train_batch(self):
        X, y = zip(*random.sample(self.buffer, self.batch_size))
        X = torch.stack(X, dim=0)
        y = torch.stack(y, dim=0)
        y_hat = self.mlp(X)

        loss = self.loss(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mlp.parameters(), 1)
        self.optimizer.step()

        return loss.item()

    def _validate(self):
        X = []
        y = []
        for _ in range(self.batch_size):
            x_sample, y_sample = self._make_data()
            X.append(x_sample)
            y.append(y_sample)
        X = torch.stack(X, dim=0)
        y = torch.stack(y, dim=0)
        with torch.no_grad():
            self.mlp.eval()
            y_hat = self.mlp(X)
            loss = self.val_loss(y_hat, y)
            self.mlp.train()
            mean_val = torch.mean(torch.abs(y))
        M1 = X[0][:np.product(self.M1_dim)].reshape(self.M1_dim)
        M2 = X[0][np.product(self.M1_dim):].reshape(self.M2_dim)
        M_out = y[0].reshape(self.out_dim)
        M_fitted = y_hat[0].reshape(self.out_dim)

        print(f"average loss per matrix element is {loss}")
        print("-" * 40)
        print("-" * 40)
        print(f"Matrix 1 is {M1}")
        print("-" * 40)
        print(f"Matrix 2 is {M2}")
        print("-" * 40)
        print(f"predicted output is {M_fitted}")
        print("-" * 40)
        print(f"reference is is {M_out}")
        print("-" * 40)
        print(f"diff is is {M_out - M_fitted}")

        return loss, mean_val

    def train(self):
        self._fill_buffer()
        self.mlp.train()
        for step in trange(1, self.n_steps+1):
            loss = self._train_batch()
            self.writer.add_scalar("Train/avg_loss", loss, step)
            self.buffer.append(self._make_data())
            if step % self.val_every == 0:
                val_loss, mean_val = self._validate()
                self.writer.add_scalar("Validate/avg_error", val_loss, step//self.val_every)
                self.writer.add_scalar("Validate/percent_off", (val_loss/mean_val)*100, step//self.val_every)
                print(f"average loss per matrix element is {val_loss}")

