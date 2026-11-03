"""
qplex_baseline.py — QPLEX (Wang et al., ICLR 2021) for E2, R1.2.

Duplex dueling mixer, same agent networks, hyperparameters, replay and epsilon schedule as
baselines/qmix_baseline.py (only the mixer changes):

    Q_i'   = w_i(s) * Q_i + b_i(s)              (w_i > 0, transformation network)
    V_i'   = w_i(s) * max_a Q_i + b_i(s)
    A_i'   = Q_i' - V_i' = w_i(s) * (Q_i - max_a Q_i)   (<= 0)
    Q_tot  = sum_i V_i' + sum_i lambda_i(s, a) * A_i'    (lambda_i > 0)

Because A_i' <= 0 and lambda_i > 0, argmax_a Q_tot = (argmax Q_1, argmax Q_2): IGM holds, while
lambda_i(s, a) depends on the JOINT action, so the class of representable Q_tot is IGM-complete
(not restricted to monotonic mixing). lambda is computed by a small multi-head attention-style
network over [s, a_1, a_2] (K heads, positive via abs + 1e-10 as in the reference implementation).
Trained on the shared team reward (sum of individual rewards), as QMIX/VDN in E2.
"""
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.qmix_baseline import AgentNet


class QPLEXMixer(nn.Module):
    def __init__(self, n_agents, state_dim, n_actions, hidden=32, heads=4):
        super().__init__()
        self.n, self.na, self.K = n_agents, n_actions, heads
        self.w = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_agents))
        self.b = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_agents))
        sa = state_dim + n_agents * n_actions
        self.key = nn.ModuleList([nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                                  for _ in range(heads)])
        self.agent = nn.ModuleList([nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_agents))
                                    for _ in range(heads)])
        self.action = nn.ModuleList([nn.Sequential(nn.Linear(sa, hidden), nn.ReLU(), nn.Linear(hidden, n_agents))
                                     for _ in range(heads)])

    def lambdas(self, state, actions_onehot):
        sa = torch.cat([state, actions_onehot.view(state.size(0), -1)], dim=1)
        lam = 0
        for k in range(self.K):
            key = torch.abs(self.key[k](state)) + 1e-10                 # [B,1]
            ag = torch.sigmoid(self.agent[k](state))                    # [B,n]
            ac = torch.sigmoid(self.action[k](sa))                      # [B,n]
            lam = lam + key * ag * ac
        return lam                                                       # [B,n] > 0

    def forward(self, chosen_qs, max_qs, state, actions_onehot):
        w = torch.abs(self.w(state)) + 1e-10
        b = self.b(state)
        v = (w * max_qs + b).sum(dim=1)
        adv = w * (chosen_qs - max_qs)                                   # <= 0
        return v + (self.lambdas(state, actions_onehot) * adv).sum(dim=1)


class QPLEXTeam:
    def __init__(self, obs_dim, state_dim, n_actions, cfg, seed=0):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        nc, lp = cfg["neural"], cfg["learning"]
        self.n_actions, self.gamma = n_actions, lp["gamma"]
        self.epsilon, self.epsilon_min, self.epsilon_decay = lp["epsilon_start"], lp["epsilon_min"], lp["epsilon_decay"]
        self.batch_size, self.target_update = nc["batch_size"], nc["target_update"]
        self.q_nets = [AgentNet(obs_dim, n_actions, nc["hidden_dim"]) for _ in range(2)]
        self.target_nets = [AgentNet(obs_dim, n_actions, nc["hidden_dim"]) for _ in range(2)]
        self.mixer = QPLEXMixer(2, state_dim, n_actions, nc["mixing_hidden"])
        self.target_mixer = QPLEXMixer(2, state_dim, n_actions, nc["mixing_hidden"])
        for i in range(2):
            self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        params = [p for n in self.q_nets for p in n.parameters()] + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=nc["lr"])
        self.buffer = deque(maxlen=nc["buffer_size"]); self.step_count = 0

    def act(self, state, obs):
        acts = []
        for i in range(2):
            if random.random() < self.epsilon:
                acts.append(random.randint(0, self.n_actions - 1))
            else:
                with torch.no_grad():
                    acts.append(int(self.q_nets[i](torch.tensor(obs[i], dtype=torch.float32)).argmax().item()))
        return tuple(acts)

    def greedy(self, obs):
        with torch.no_grad():
            return tuple(int(self.q_nets[i](torch.tensor(obs[i], dtype=torch.float32)).argmax().item()) for i in range(2))

    def _onehot(self, a):
        return F.one_hot(a, self.n_actions).float()

    def learn(self, state_vec, actions, rewards, next_state_vec, obs, next_obs, done):
        self.buffer.append((obs.copy(), actions, float(np.sum(rewards)), next_obs.copy(),
                            state_vec.copy(), next_state_vec.copy(), done))
        self.step_count += 1
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        f = lambda k, dt=torch.float32: torch.tensor(np.array([b[k] for b in batch]), dtype=dt)
        obs_b, act_b, rew_b, nobs_b, st_b, nst_b, done_b = f(0), f(1, torch.long), f(2), f(3), f(4), f(5), f(6)
        B = self.batch_size
        chosen, maxq = torch.zeros(B, 2), torch.zeros(B, 2)
        for i in range(2):
            q = self.q_nets[i](obs_b[:, i, :])
            chosen[:, i] = q.gather(1, act_b[:, i:i + 1]).squeeze(1)
            maxq[:, i] = q.max(dim=1)[0].detach()
        q_tot = self.mixer(chosen, maxq, st_b, self._onehot(act_b))
        with torch.no_grad():
            na, tq, tmax = torch.zeros(B, 2, dtype=torch.long), torch.zeros(B, 2), torch.zeros(B, 2)
            for i in range(2):
                na[:, i] = self.q_nets[i](nobs_b[:, i, :]).argmax(dim=1)        # double-Q action selection
                qn = self.target_nets[i](nobs_b[:, i, :])
                tq[:, i] = qn.gather(1, na[:, i:i + 1]).squeeze(1)
                tmax[:, i] = qn.max(dim=1)[0]
            target = rew_b + self.gamma * (1 - done_b) * self.target_mixer(tq, tmax, nst_b, self._onehot(na))
        loss = F.mse_loss(q_tot, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.step_count % self.target_update == 0:
            for i in range(2):
                self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
