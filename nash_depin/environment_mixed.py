"""
environment_mixed.py — Gridworld with MIXED-MOTIVE rewards for E2.
Nash-DePIN.

Key difference from E1: this environment has a mixed-motive reward structure
combining:
  - COMPETITIVE:  +10 for completing a task (only one robot gets it)
  - COOPERATIVE:  -5 for a collision, +2 for spatial cooperation

This is precisely the structure QMIX/VDN cannot represent well (due to their
monotonicity/additivity constraint over a shared reward), while Nash
Q-Learning can, because each agent optimizes its own payoff.

The environment ALWAYS returns PER-AGENT rewards. Each algorithm decides how
to use them:
  - Nash:      uses each agent's individual reward
  - QMIX/VDN:  use the SUM (team reward) — they are purely cooperative
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Task:
    task_id: int
    position: Tuple[int, int]
    reward: float = 10.0
    completed: bool = False
    assigned_to: Optional[int] = None


class MixedMotiveGridWorld:
    """
    NxN gridworld, 2 agents, mixed-motive rewards.

    Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
    Per-agent observation: [own_row, own_col, other_row, other_col],
        normalized to [0,1] — a 4-dimensional vector.
    Global state (for QMIX mixing and Nash): integer encoding of positions.
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
    N_ACTIONS = 5
    OBS_DIM = 4

    def __init__(
        self,
        grid_size: int = 5,
        n_agents: int = 2,
        n_tasks: int = 4,
        task_reward: float = 10.0,
        collision_penalty: float = -5.0,
        cooperation_reward: float = 2.0,
        coop_distance: int = 1,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.task_reward = task_reward
        self.collision_penalty = collision_penalty
        self.cooperation_reward = cooperation_reward
        self.coop_distance = coop_distance
        self.max_steps = max_steps

        self.n_states = grid_size ** (2 * n_agents)
        self.rng = np.random.default_rng(seed)

        self.agent_positions: List[Tuple[int, int]] = []
        self.tasks: List[Task] = []
        self.step_count = 0
        self._setup_task_positions()

    def _setup_task_positions(self) -> None:
        """Fixed task positions at the grid corners and edge midpoints."""
        g = self.grid_size - 1
        candidates = [(0, 0), (0, g), (g, 0), (g, g),
                      (0, g // 2), (g // 2, 0), (g, g // 2), (g // 2, g)]
        self._task_positions = candidates[:self.n_tasks]

    # ------------------------------------------------------------------

    def reset(self) -> Tuple[int, np.ndarray]:
        """
        Resets the environment.
        Returns:
            state : int (global state, for Nash and the mixing network)
            obs   : array [n_agents, OBS_DIM] (per-agent observations)
        """
        starts = [(0, 0), (self.grid_size - 1, self.grid_size - 1),
                  (0, self.grid_size - 1), (self.grid_size - 1, 0)]
        self.agent_positions = [starts[i] for i in range(self.n_agents)]
        self.tasks = [
            Task(task_id=i, position=self._task_positions[i], reward=self.task_reward)
            for i in range(self.n_tasks)
        ]
        self.step_count = 0
        return self._encode_state(), self._get_obs()

    def step(self, actions: Tuple[int, ...]) -> Tuple[int, np.ndarray, List[float], bool, Dict]:
        """
        Executes a joint action.

        Returns:
            next_state : int (global state)
            next_obs   : array [n_agents, OBS_DIM]
            rewards    : list of PER-AGENT rewards
            done       : bool (episode ends at the step limit)
            info       : dict with collisions, tasks completed, cooperations
        """
        self.step_count += 1
        rewards = [0.0] * self.n_agents
        info = {"collisions": 0, "tasks_completed": 0, "cooperations": 0}

        # 1) New positions (clipped to the grid bounds)
        new_positions = []
        for i, action in enumerate(actions):
            dr, dc = self.ACTIONS[action]
            r, c = self.agent_positions[i]
            nr = max(0, min(self.grid_size - 1, r + dr))
            nc = max(0, min(self.grid_size - 1, c + dc))
            new_positions.append((nr, nc))

        # 2) COLLISIONS (cooperative component): penalize both, revert move
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if new_positions[i] == new_positions[j]:
                    rewards[i] += self.collision_penalty
                    rewards[j] += self.collision_penalty
                    info["collisions"] += 1
                    new_positions[i] = self.agent_positions[i]
                    new_positions[j] = self.agent_positions[j]

        self.agent_positions = new_positions

        # 3) TASKS (competitive component): only ONE robot gets each task
        for task in self.tasks:
            if task.completed:
                continue
            for i, pos in enumerate(self.agent_positions):
                if pos == task.position:
                    rewards[i] += task.reward
                    task.completed = True
                    task.assigned_to = i
                    info["tasks_completed"] += 1
                    break

        # 4) SPATIAL COOPERATION (cooperative component): if the agents are
        #    within coop_distance WITHOUT colliding, both get +cooperation_reward
        if self.n_agents == 2 and info["collisions"] == 0:
            (r0, c0), (r1, c1) = self.agent_positions[0], self.agent_positions[1]
            manhattan = abs(r0 - r1) + abs(c0 - c1)
            if 0 < manhattan <= self.coop_distance:
                rewards[0] += self.cooperation_reward
                rewards[1] += self.cooperation_reward
                info["cooperations"] += 1

        # 5) Respawn completed tasks at a random position from the candidate set
        #    (continuing environment)
        for task in self.tasks:
            if task.completed:
                task.completed = False
                task.assigned_to = None
                task.position = self._task_positions[
                    self.rng.integers(0, len(self._task_positions))
                ]

        done = self.step_count >= self.max_steps
        return self._encode_state(), self._get_obs(), rewards, done, info

    # ------------------------------------------------------------------

    def _encode_state(self) -> int:
        """Encodes all agent positions into a single integer state."""
        state = 0
        for r, c in self.agent_positions:
            state = state * self.grid_size * self.grid_size + r * self.grid_size + c
        return state % self.n_states

    def _get_obs(self) -> np.ndarray:
        """Normalized per-agent observation: [own_row, own_col, other_row, other_col]."""
        g = self.grid_size - 1
        obs = np.zeros((self.n_agents, self.OBS_DIM), dtype=np.float32)
        for i in range(self.n_agents):
            r, c = self.agent_positions[i]
            j = (i + 1) % self.n_agents
            ro, co = self.agent_positions[j]
            obs[i] = [r / g, c / g, ro / g, co / g]
        return obs

    @property
    def state_dim(self) -> int:
        """Global state dimension (for the QMIX mixing network)."""
        return 2 * self.n_agents  # normalized (row, col) per agent

    def get_global_state_vec(self) -> np.ndarray:
        """Global state as a continuous vector (for the QMIX mixing network)."""
        g = self.grid_size - 1
        vec = []
        for r, c in self.agent_positions:
            vec.extend([r / g, c / g])
        return np.array(vec, dtype=np.float32)

    @property
    def arena_description(self) -> str:
        return (f"{self.grid_size}x{self.grid_size} mixed-motive | {self.n_agents} agents | "
                f"{self.n_tasks} tasks | +{self.task_reward}/task "
                f"{self.collision_penalty}/collision +{self.cooperation_reward}/coop")
