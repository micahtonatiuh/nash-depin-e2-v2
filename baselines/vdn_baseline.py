"""
vdn_baseline.py — Value Decomposition Networks (Sunehag et al. 2017)
Nash-DePIN — cooperative baseline for E2.

Core idea:
    Q_tot(s, a) = sum_i Q_i(o_i, a_i)      (ADDITIVE decomposition)

trained on a SHARED team reward:
    r_team = sum_i r_i

Why this causes monopolization (high Gini) on mixed-motive tasks:
VDN assumes the team value is the sum of individual values. Since the reward
is shared, each agent only maximizes the total — it is indifferent to whether
it or the other agent completes the task. The typical outcome is that one
agent takes the tasks while the other avoids collisions, because that
maximizes the sum. This raises the Gini coefficient.

Reference: Sunehag et al., "Value-Decomposition Networks For Cooperative
Multi-Agent Learning", 2017.
"""

import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentNet(nn.Module):
    """Per-agent Q network: observation -> Q value per action."""

    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.out(x)


class VDNTeam:
    """
    VDN team: 2 agents with individual Q networks, additive team value.
    Trains on a SHARED reward (the sum of individual rewards).
    """

    def __init__(self, obs_dim, n_actions, cfg, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        nn_cfg = cfg["neural"]
        lp = cfg["learning"]
        self.n_actions = n_actions
        self.n_agents = 2
        self.gamma = lp["gamma"]
        self.epsilon = lp["epsilon_start"]
        self.epsilon_min = lp["epsilon_min"]
        self.epsilon_decay = lp["epsilon_decay"]
        self.batch_size = nn_cfg["batch_size"]
        self.target_update = nn_cfg["target_update"]

        # One Q network per agent (no weight sharing, so agents can specialize)
        self.q_nets = [AgentNet(obs_dim, n_actions, nn_cfg["hidden_dim"]) for _ in range(2)]
        self.target_nets = [AgentNet(obs_dim, n_actions, nn_cfg["hidden_dim"]) for _ in range(2)]
        for i in range(2):
            self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())

        params = [p for net in self.q_nets for p in net.parameters()]
        self.optimizer = torch.optim.Adam(params, lr=nn_cfg["lr"])

        self.buffer = deque(maxlen=nn_cfg["buffer_size"])
        self.step_count = 0

    def act(self, state, obs):
        """epsilon-greedy over each agent Q network."""
        actions = []
        for i in range(2):
            if random.random() < self.epsilon:
                actions.append(random.randint(0, self.n_actions - 1))
            else:
                with torch.no_grad():
                    q = self.q_nets[i](torch.tensor(obs[i], dtype=torch.float32))
                    actions.append(int(q.argmax().item()))
        return tuple(actions)

    def learn(self, state, actions, rewards, next_state, obs, next_obs, done):
        """
        Stores the transition and trains.
        Key point: uses the TEAM reward (sum) — VDN is purely cooperative.
        """
        team_reward = float(np.sum(rewards))   # shared reward
        self.buffer.append((obs.copy(), actions, team_reward, next_obs.copy(), done))
        self.step_count += 1

        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        obs_b = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        act_b = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.long)
        rew_b = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32)
        nobs_b = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        done_b = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32)

        # Q_tot = sum_i Q_i(o_i, a_i)  -> ADDITIVE decomposition
        q_tot = torch.zeros(self.batch_size)
        for i in range(2):
            q_i = self.q_nets[i](obs_b[:, i, :])
            q_tot += q_i.gather(1, act_b[:, i:i+1]).squeeze(1)

        # Target: r_team + gamma * sum_i max_a Q_target_i(o'_i, a)
        with torch.no_grad():
            q_tot_next = torch.zeros(self.batch_size)
            for i in range(2):
                q_next = self.target_nets[i](nobs_b[:, i, :])
                q_tot_next += q_next.max(dim=1)[0]
            target = rew_b + self.gamma * q_tot_next * (1 - done_b)

        loss = F.mse_loss(q_tot, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_count % self.target_update == 0:
            for i in range(2):
                self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
