"""SAC actor/critic 网络定义 — train_sac.py 与 eval_sac.py 共用.

放在顶层模块 (而不是 __main__) 是为了让 agent pickle 在 train/eval 两端
都能解析到同一个类路径 `networks.ActorNetwork`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features=256, **_):
        super().__init__()
        self._h1 = nn.Linear(input_shape[0], n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, output_shape[0])
        for l in (self._h1, self._h2, self._out):
            nn.init.xavier_uniform_(l.weight)

    def forward(self, x, **_):
        x = x.float()
        h = F.relu(self._h1(x))
        h = F.relu(self._h2(h))
        return self._out(h)


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, action_dim, n_features=256, **_):
        super().__init__()
        self._h1 = nn.Linear(input_shape[0] + action_dim, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, output_shape[0])
        for l in (self._h1, self._h2, self._out):
            nn.init.xavier_uniform_(l.weight)

    def forward(self, state, action, **_):
        state = state.float()
        h = F.relu(self._h1(torch.cat([state, action.float()], dim=-1)))
        h = F.relu(self._h2(h))
        return self._out(h).squeeze(-1)
