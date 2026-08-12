"""
nash_agent_e2.py — Nash Q-Learning for E2 (PER-AGENT rewards).
Nash-DePIN.

Reuses the bimatrix solver validated in E1 (joint_nash), now with the
egalitarian refinement enabled (equity=True). Each agent keeps its own Q-table
and receives its own reward. When the stage game admits several Nash
equilibria, the solver selects the most equitable one (smallest |V0 - V1|),
steering learning toward task-rotation equilibria rather than monopoly
equilibria.
"""

import numpy as np
from typing import List, Tuple
from nash_depin.nash_solver import joint_nash

# Egalitarian refinement: select the most equitable equilibrium.
# This flag was TRUE for all E2 results reported in the paper.
USE_EQUITY = True


class NashQAgentE2:
    """Tabular Nash Q-Learning agent with an individual reward signal."""

    def __init__(self, agent_id, n_agents, n_states, n_actions,
                 alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions, n_actions))
        self.policy = np.ones((n_states, n_actions)) / n_actions

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.random.choice(self.n_actions, p=self.policy[state]))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class NashTeamE2:
    """
    Coordinates 2 Nash agents with egalitarian equilibrium refinement.
    Trains on the mixed-motive environment using individual rewards.
    """

    def __init__(self, n_states, n_actions, cfg, seed=0):
        np.random.seed(seed)
        lp = cfg["learning"]
        self.agents = [
            NashQAgentE2(
                agent_id=i, n_agents=2, n_states=n_states, n_actions=n_actions,
                alpha=lp["alpha"], gamma=lp["gamma"],
                epsilon=lp["epsilon_start"], epsilon_min=lp["epsilon_min"],
                epsilon_decay=lp["epsilon_decay"],
            )
            for i in range(2)
        ]
        self.n_actions = n_actions

    def act(self, state: int, obs=None) -> Tuple[int, ...]:
        return tuple(a.select_action(state) for a in self.agents)

    def learn(self, state, actions, rewards, next_state, obs=None, next_obs=None, done=False):
        """
        Nash-Q update with INDIVIDUAL rewards and egalitarian refinement.
        A single joint_nash call per step (fast=True, equity=True) is reused
        for both the target value and the policy.
        """
        q0, q1 = self.agents[0].q_table, self.agents[1].q_table
        # Egalitarian equilibrium of the next state (value + policy)
        s0n, s1n, V0, V1 = joint_nash(q0[next_state], q1[next_state],
                                      fast=True, equity=USE_EQUITY)
        nash_next = [V0, V1]

        for i, agent in enumerate(self.agents):
            q = agent.q_table[state, actions[0], actions[1]]
            target = rewards[i] + agent.gamma * nash_next[i]
            agent.q_table[state, actions[0], actions[1]] += agent.alpha * (target - q)

        # Reuse the next_state policy already computed above
        self.agents[0].policy[next_state] = s0n
        self.agents[1].policy[next_state] = s1n
        # Recompute the current-state policy (its Q changed this step)
        s0, s1, _, _ = joint_nash(q0[state], q1[state], fast=True, equity=USE_EQUITY)
        self.agents[0].policy[state] = s0
        self.agents[1].policy[state] = s1

    def end_episode(self):
        for a in self.agents:
            a.decay_epsilon()

    @property
    def epsilon(self):
        return self.agents[0].epsilon
