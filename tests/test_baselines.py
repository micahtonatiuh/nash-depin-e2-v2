"""
test_baselines.py — Unit tests for E2.
Run: pytest tests/test_baselines.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nash_depin.environment_mixed import MixedMotiveGridWorld
from nash_depin.fairness_metrics import (
    gini_coefficient, jain_fairness_index, task_balance_ratio
)


# ================================================================
# Fairness metrics
# ================================================================

def test_gini_perfect_equality():
    """Gini = 0 when all agents receive the same."""
    assert gini_coefficient([5.0, 5.0]) < 1e-6

def test_gini_max_inequality():
    """Gini is high when one agent takes everything."""
    assert gini_coefficient([10.0, 0.0]) > 0.4

def test_gini_range():
    """Gini always lies in [0, 1]."""
    for vals in [[1, 2], [10, 0], [3, 3], [100, 1]]:
        g = gini_coefficient(vals)
        assert 0.0 <= g <= 1.0

def test_gini_handles_negative():
    """Gini handles negative rewards without breaking."""
    g = gini_coefficient([-5.0, 10.0])
    assert 0.0 <= g <= 1.0

def test_jain_perfect_equality():
    """Jain = 1 when all agents are equal."""
    assert abs(jain_fairness_index([5.0, 5.0]) - 1.0) < 1e-6

def test_jain_min_for_two_agents():
    """Jain -> 0.5 when one agent has everything (n=2)."""
    assert abs(jain_fairness_index([10.0, 0.0]) - 0.5) < 0.01

def test_jain_range():
    """Jain lies in [1/n, 1]."""
    for vals in [[1, 2], [10, 0], [3, 3]]:
        j = jain_fairness_index(vals)
        assert 0.5 <= j <= 1.0

def test_task_balance_perfect():
    """Balance = 1 when both agents complete the same number of tasks."""
    assert task_balance_ratio([5, 5]) == 1.0

def test_task_balance_monopoly():
    """Balance = 0 when one agent takes all the tasks."""
    assert task_balance_ratio([10, 0]) == 0.0


# ================================================================
# Mixed-motive environment
# ================================================================

@pytest.fixture
def env():
    return MixedMotiveGridWorld(grid_size=5, n_agents=2, n_tasks=4, seed=0)

def test_env_reset_returns_state_and_obs(env):
    state, obs = env.reset()
    assert isinstance(state, int)
    assert obs.shape == (2, 4)

def test_env_obs_normalized(env):
    """Observations are normalized to [0,1]."""
    _, obs = env.reset()
    assert obs.min() >= 0.0 and obs.max() <= 1.0

def test_env_step_returns_per_agent_rewards(env):
    """step returns PER-AGENT rewards (a list of 2)."""
    env.reset()
    _, _, rewards, _, _ = env.step((4, 4))
    assert len(rewards) == 2

def test_env_collision_penalty(env):
    """A collision penalizes both agents."""
    env.reset()
    env.agent_positions = [(2, 2), (2, 3)]
    _, _, rewards, _, info = env.step((3, 2))  # both move toward the same cell
    if info["collisions"] > 0:
        assert rewards[0] < 0 and rewards[1] < 0

def test_env_task_reward_single_agent(env):
    """Only one agent receives the reward for a given task."""
    env.reset()
    task_pos = env.tasks[0].position
    env.agent_positions[0] = task_pos
    env.agent_positions[1] = (2, 2)
    _, _, rewards, _, info = env.step((4, 4))
    # If a task was completed, exactly one agent received +task_reward
    if info["tasks_completed"] > 0:
        got = sum(1 for r in rewards if r >= env.task_reward)
        assert got == 1

def test_env_global_state_vec(env):
    """The global state vector has the expected dimension."""
    env.reset()
    vec = env.get_global_state_vec()
    assert vec.shape == (env.state_dim,)

def test_env_done_at_max_steps(env):
    """The episode ends when max_steps is reached."""
    env.reset()
    done = False
    steps = 0
    while not done and steps < env.max_steps + 5:
        _, _, _, done, _ = env.step((4, 4))
        steps += 1
    assert done
    assert steps == env.max_steps


# ================================================================
# Neural baselines (smoke tests — require torch)
# ================================================================

def test_vdn_imports_and_acts():
    """VDN builds and produces valid actions."""
    torch = pytest.importorskip("torch")
    from baselines.vdn_baseline import VDNTeam
    cfg = {
        "neural": {"lr": 0.0005, "hidden_dim": 32, "mixing_hidden": 16,
                   "buffer_size": 100, "batch_size": 8, "target_update": 50},
        "learning": {"gamma": 0.95, "epsilon_start": 1.0,
                     "epsilon_min": 0.05, "epsilon_decay": 0.995},
    }
    team = VDNTeam(obs_dim=4, n_actions=5, cfg=cfg, seed=0)
    obs = np.zeros((2, 4), dtype=np.float32)
    actions = team.act(0, obs)
    assert len(actions) == 2
    assert all(0 <= a < 5 for a in actions)

def test_qmix_imports_and_acts():
    """QMIX builds and produces valid actions."""
    torch = pytest.importorskip("torch")
    from baselines.qmix_baseline import QMIXTeam
    cfg = {
        "neural": {"lr": 0.0005, "hidden_dim": 32, "mixing_hidden": 16,
                   "buffer_size": 100, "batch_size": 8, "target_update": 50},
        "learning": {"gamma": 0.95, "epsilon_start": 1.0,
                     "epsilon_min": 0.05, "epsilon_decay": 0.995},
    }
    team = QMIXTeam(obs_dim=4, state_dim=4, n_actions=5, cfg=cfg, seed=0)
    obs = np.zeros((2, 4), dtype=np.float32)
    actions = team.act(0, obs)
    assert len(actions) == 2

def test_qmix_mixing_monotonic():
    """The QMIX mixing network uses non-negative weights (monotonicity)."""
    torch = pytest.importorskip("torch")
    from baselines.qmix_baseline import MixingNetwork
    mixer = MixingNetwork(n_agents=2, state_dim=4, mixing_hidden=16)
    state = torch.zeros(1, 4)
    # Q_tot must not decrease when any Q_i increases
    q_low = torch.tensor([[1.0, 1.0]])
    q_high = torch.tensor([[5.0, 1.0]])
    with torch.no_grad():
        tot_low = mixer(q_low, state).item()
        tot_high = mixer(q_high, state).item()
    assert tot_high >= tot_low - 1e-4   # monotonicity


def test_nash_team_acts():
    """The E2 Nash team produces valid actions."""
    from nash_depin.nash_agent_e2 import NashTeamE2
    cfg = {"learning": {"alpha": 0.1, "gamma": 0.95, "epsilon_start": 1.0,
                        "epsilon_min": 0.05, "epsilon_decay": 0.995}}
    team = NashTeamE2(n_states=625, n_actions=5, cfg=cfg, seed=0)
    actions = team.act(0)
    assert len(actions) == 2
    assert all(0 <= a < 5 for a in actions)
