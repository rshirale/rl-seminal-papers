"""Tests for the Chapter 5 notebook.

The notebook re-implements PPO inline so readers can follow the derivation cell
by cell. That duplication is a teaching choice, but it means the notebook can
drift away from the module files it mirrors -- and it did, in both directions
at once. The notebook's ``Actor`` matched listing 5.1 while ``actor_critic.py``
still carried a ``SquashedNormal``, so whichever copy a reader trusted, the
other one disagreed with it. Separately the notebook was re-drawing its
minibatch permutation inside the inner loop and zeroing the GAE bootstrap at
every time-limit truncation. Chapters 3 and 4 have had a notebook test for
exactly this class of problem; chapter 5 did not, which is why none of it
surfaced.

These tests cover both risks:

  * parity    -- the notebook's inline classes still behave like the modules
  * execution -- the notebook actually runs top to bottom without raising

The execution test is marked ``slow`` because it runs a real training loop.
Run everything with ``make test-all``; the default ``make test`` skips it.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

CH5 = (
    Path(__file__).resolve().parents[1]
    / "src" / "part_2_methods" / "ch05_ppo"
)
NOTEBOOK = CH5 / "Chapter5_PPO.ipynb"


def code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def find_cell(marker):
    """The first code cell containing `marker`. Located by content rather than
    index so inserting a cell does not silently break these tests."""
    for source in code_cells():
        if marker in source:
            return source
    raise AssertionError(f"No notebook code cell contains {marker!r}")


def executable(cell):
    """The cell with comments stripped.

    Prose and commented-out Colab lines may legitimately name things the
    executable code must not.
    """
    return "\n".join(line.split("#")[0] for line in cell.splitlines())


@pytest.fixture(scope="module")
def notebook_ns():
    """Executes the notebook's definition cells (not its training loop) and
    returns the resulting namespace."""
    import matplotlib
    matplotlib.use("Agg")  # no GUI during tests

    namespace = {"__name__": "__notebook__"}
    for marker in ("import gymnasium", "class Actor", "class PPOAgent"):
        exec(compile(find_cell(marker), "<notebook>", "exec"), namespace)
    return namespace


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

def test_notebook_is_valid_json_and_has_cells():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 0


def test_notebook_defines_the_classes_the_chapter_describes(notebook_ns):
    for name in ("Actor", "Critic", "PPOAgent"):
        assert name in notebook_ns, f"notebook no longer defines {name}"


def test_agent_is_constructible(notebook_ns):
    """The cheapest possible guard against a paste error in the agent."""
    agent = notebook_ns["PPOAgent"](state_dim=3, action_dim=1, max_action=2.0)
    action, logprob, value = agent.select_action(np.zeros(3, dtype=np.float32))
    assert action.shape == (1,)
    assert np.isfinite(logprob) and np.isfinite(value)


def test_training_cell_episode_count_is_overridable():
    """Guards the hook the execution test depends on."""
    assert "CH5_NUM_EPISODES" in find_cell("MAX_EPISODES")


def test_colab_install_pins_match_the_requirements_files():
    """The Colab cell is the only install path a reader on Colab ever sees, so
    it must not drift from the repo's own bounds.

    Regression: it pinned ``gymnasium>=0.29.1,<1.1`` against the repo's
    ``>=1.0,<2.0``, unquoted, so the shell also ate the ``>=`` as a redirect --
    the identical defect chapter 4's notebook shipped with.
    """
    install = find_cell("%pip install")
    assert '"gymnasium[classic-control]>=1.0,<2.0"' in install
    assert '"torch>=2.0.0"' in install


# --------------------------------------------------------------------------
# Parity with the module implementations
# --------------------------------------------------------------------------

def test_notebook_actor_matches_module_architecture(notebook_ns):
    from src.part_2_methods.ch05_ppo import Actor as ModuleActor
    from src.part_2_methods.ch05_ppo import Critic as ModuleCritic

    nb_actor, mod_actor = notebook_ns["Actor"](3, 1, 2.0), ModuleActor(3, 1, 2.0)
    nb_critic, mod_critic = notebook_ns["Critic"](3), ModuleCritic(3)

    def shapes(net):
        return [(m.in_features, m.out_features)
                for m in net.modules() if isinstance(m, torch.nn.Linear)]

    assert shapes(nb_actor) == shapes(mod_actor)
    assert shapes(nb_critic) == shapes(mod_critic)
    assert nb_actor.log_std.shape == mod_actor.log_std.shape


def test_notebook_actor_returns_a_plain_normal(notebook_ns):
    """Listing 5.1 teaches ``torch.distributions.Normal`` with only the mean
    squashed. The module carried a ``SquashedNormal`` for a while; if either
    copy reintroduces one, the log-probabilities stop matching the text.
    """
    from torch.distributions import Normal

    dist = notebook_ns["Actor"](3, 1, 2.0)(torch.zeros(4, 3))
    assert type(dist) is Normal
    # The mean is bounded by the tanh; the samples deliberately are not.
    assert torch.all(dist.mean.abs() <= 2.0)


def test_notebook_actor_matches_module_distribution(notebook_ns):
    """Same weights in, same distribution out."""
    from src.part_2_methods.ch05_ppo import Actor as ModuleActor

    nb_actor = notebook_ns["Actor"](3, 1, 2.0)
    mod_actor = ModuleActor(3, 1, 2.0)
    mod_actor.load_state_dict(nb_actor.state_dict())

    states = torch.randn(8, 3)
    nb_dist, mod_dist = nb_actor(states), mod_actor(states)
    assert torch.allclose(nb_dist.mean, mod_dist.mean)
    assert torch.allclose(nb_dist.stddev, mod_dist.stddev)


def test_notebook_minibatches_partition_the_epoch():
    """Regression: the index was ``np.random.permutation(n)[start:stop]``,
    re-drawn inside the inner loop, so every minibatch resampled the whole
    rollout instead of partitioning one epoch across it."""
    code = executable(find_cell("class PPOAgent"))
    assert "perm = np.random.permutation(n)" in code
    assert "np.random.permutation(n)[start" not in code


def test_notebook_training_loop_bootstraps_through_truncation():
    """Regression: storing ``float(done or truncated)`` zeroed the GAE
    bootstrap at Pendulum's 200-step time limit -- on every episode, since
    Pendulum-v1 never terminates."""
    code = executable(find_cell("MAX_EPISODES"))
    assert "float(done or truncated)" not in code
    assert "float(done)" in code


def test_notebook_agent_clips_the_ratio_and_reports_diagnostics():
    """The clipped objective and the two metrics section 7 teaches."""
    code = executable(find_cell("class PPOAgent"))
    assert "torch.clamp(" in code
    assert "torch.min(surr1, surr2)" in code
    assert "approx_kl" in code and "clip_frac" in code


def test_notebook_agent_can_learn_on_pendulum(notebook_ns):
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    torch.manual_seed(0)
    np.random.seed(0)
    agent = notebook_ns["PPOAgent"](
        state_dim=3, action_dim=1, max_action=2.0, batch_size=16, k_epochs=2)

    state, _ = env.reset(seed=0)
    rollouts = []
    for _ in range(64):
        action, logprob, value = agent.select_action(state)
        nxt, reward, terminated, truncated, _ = env.step(action)
        rollouts.append((state, action, reward, nxt, float(terminated),
                         logprob, value))
        state = nxt if not (terminated or truncated) else env.reset()[0]

    before = [p.detach().clone() for p in agent.actor.parameters()]
    approx_kl, clip_frac = agent.update(rollouts)
    after = list(agent.actor.parameters())

    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    assert np.isfinite(approx_kl)
    assert 0.0 <= clip_frac <= 1.0
    env.close()


# --------------------------------------------------------------------------
# Full execution
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_notebook_runs_top_to_bottom(monkeypatch):
    """Executes every code cell in order, exactly as a reader would.

    Runs in-process rather than through a Jupyter kernel on purpose: a kernel
    would resolve the ``python3`` kernelspec, which can point at a different
    interpreter than the one running the tests -- silently validating some
    other environment's packages. Every magic in this notebook is commented
    out, so in-process execution covers the same ground.
    """
    import matplotlib
    matplotlib.use("Agg")

    monkeypatch.setenv("CH5_NUM_EPISODES", "40")
    monkeypatch.chdir(CH5)

    namespace = {"__name__": "__notebook__"}
    for index, source in enumerate(code_cells()):
        try:
            exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)
        except Exception as exc:
            pytest.fail(
                f"Notebook cell {index} raised {type(exc).__name__}: {exc}\n"
                f"--- cell source ---\n{source[:600]}"
            )

    assert len(namespace["ep_rewards"]) == 40
    assert all(r < 0 for r in namespace["ep_rewards"])  # Pendulum rewards are negative
    # The diagnostics the chapter teaches must actually be populated.
    assert namespace["kl_history"] and namespace["cf_history"]
