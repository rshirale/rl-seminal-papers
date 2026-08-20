"""Tests for the Chapter 4 notebook.

The notebook re-implements DDPG inline so readers can follow the derivation
cell by cell. That duplication is a teaching choice, but it means the notebook
can drift away from the module files it mirrors -- and it did: the shipped
notebook's ``DDPGAgent.__init__`` called a ``reset_parameters`` that referenced
``self.l3``, so constructing an agent raised ``AttributeError`` and the
training cell could never run. A peer reviewer found that, not a test.

These tests cover both risks:

  * parity    -- the notebook's inline classes still behave like the modules
  * execution -- the notebook actually runs top to bottom without raising

The execution test is marked ``slow`` because it runs a real training loop.
Run everything with ``make test-all``; the default ``make test`` skips it.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

CH4 = (
    Path(__file__).resolve().parents[1]
    / "src" / "part_2_methods" / "ch04_ddpg"
)
NOTEBOOK = CH4 / "Chapter4_DDPG.ipynb"


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


@pytest.fixture(scope="module")
def notebook_ns():
    """Executes the notebook's definition cells (not its training loop) and
    returns the resulting namespace."""
    import matplotlib
    matplotlib.use("Agg")  # no GUI during tests

    namespace = {"__name__": "__notebook__"}
    for marker in ("import gymnasium", "class Actor", "class Critic",
                   "class GaussianNoise", "class ReplayBuffer",
                   "class DDPGAgent"):
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
    for name in ("Actor", "Critic", "GaussianNoise", "ReplayBuffer", "DDPGAgent"):
        assert name in notebook_ns, f"notebook no longer defines {name}"


def test_agent_is_constructible():
    """Regression: the shipped notebook's DDPGAgent could not be constructed.

    ``__init__`` called ``reset_parameters()``, which had been pasted in from
    the network classes and referenced ``self.l3``. Every cell from the
    training loop onward raised. This is the cheapest possible guard against
    that class of paste error returning.
    """
    namespace = {"__name__": "__notebook__"}
    for marker in ("import gymnasium", "class Actor", "class Critic",
                   "class GaussianNoise", "class ReplayBuffer",
                   "class DDPGAgent"):
        exec(compile(find_cell(marker), "<notebook>", "exec"), namespace)

    agent = namespace["DDPGAgent"](state_dim=3, action_dim=1, max_action=2.0)
    action = agent.select_action(np.zeros(3, dtype=np.float32), explore=False)
    assert action.shape == (1,)


def test_training_cell_episode_count_is_overridable():
    """Guards the hook the execution test depends on."""
    assert "CH4_NUM_EPISODES" in find_cell("for ep in range(1, EPISODES + 1)")


def test_colab_install_pins_match_the_requirements_files():
    """The Colab cell is the only install path a reader on Colab ever sees, so
    it must not drift from the repo's own bounds.

    Regression: it pinned ``gymnasium>=0.29.1,<1.1`` against the repo's
    ``>=1.0,<2.0``, unquoted, so the shell also ate the ``>=`` as a redirect.
    """
    install = find_cell("%pip install")
    assert '"gymnasium[classic-control]>=1.0,<2.0"' in install
    assert '"torch>=2.0.0"' in install


# --------------------------------------------------------------------------
# Parity with the module implementations
# --------------------------------------------------------------------------

def test_notebook_networks_match_module_architecture(notebook_ns):
    from src.part_2_methods.ch04_ddpg import Actor as ModuleActor
    from src.part_2_methods.ch04_ddpg import Critic as ModuleCritic

    nb_actor, mod_actor = notebook_ns["Actor"](3, 1, 2.0), ModuleActor(3, 1, 2.0)
    nb_critic, mod_critic = notebook_ns["Critic"](3, 1), ModuleCritic(3, 1)

    for nb_net, mod_net in ((nb_actor, mod_actor), (nb_critic, mod_critic)):
        for layer in ("l1", "l2", "l3"):
            assert (getattr(nb_net, layer).in_features
                    == getattr(mod_net, layer).in_features)
            assert (getattr(nb_net, layer).out_features
                    == getattr(mod_net, layer).out_features)

    assert nb_actor(torch.zeros(4, 3)).shape == mod_actor(torch.zeros(4, 3)).shape
    assert (nb_critic(torch.zeros(4, 3), torch.zeros(4, 1)).shape
            == mod_critic(torch.zeros(4, 3), torch.zeros(4, 1)).shape)


def test_notebook_networks_apply_the_papers_output_init(notebook_ns):
    """The chapter's prose promises this; both copies must deliver it."""
    actor = notebook_ns["Actor"](3, 1, 2.0)
    critic = notebook_ns["Critic"](3, 1)
    assert actor.l3.weight.abs().max() <= 3e-3
    assert critic.l3.weight.abs().max() <= 3e-3


def test_notebook_replay_buffer_matches_module_behaviour(notebook_ns):
    from src.part_2_methods.ch04_ddpg import ReplayBuffer as ModuleBuffer

    def sampled(cls):
        buf = cls(16)
        for i in range(16):
            buf.push(np.full(3, i, dtype=np.float32),
                     np.full(1, -i, dtype=np.float32),
                     float(i), np.zeros(3, dtype=np.float32), float(i % 2))
        import random
        random.seed(0)  # both call random.sample, so the seeds align
        return [t.tolist() for t in buf.sample(6)]

    assert sampled(ModuleBuffer) == sampled(notebook_ns["ReplayBuffer"])


def test_notebook_gaussian_noise_matches_module_schedule(notebook_ns):
    from src.part_2_methods.ch04_ddpg import GaussianNoise as ModuleNoise

    def schedule(cls):
        noise = cls(1, sigma=0.2, sigma_final=0.05, decay_steps=100)
        trace = []
        for _ in range(150):
            trace.append(round(noise.sigma, 6))
            noise.sample()
        return trace

    assert schedule(ModuleNoise) == schedule(notebook_ns["GaussianNoise"])


def test_notebook_uses_as_tensor_not_legacy_constructors():
    """Regression: the modules moved to torch.as_tensor / torch.from_numpy but
    the notebook kept torch.FloatTensor, which always copies."""
    for marker in ("class DDPGAgent", "class ReplayBuffer"):
        cell = find_cell(marker)
        # Prose may name the legacy constructors; executable code may not.
        code = "\n".join(line.split("#")[0] for line in cell.splitlines())
        assert "torch.FloatTensor" not in code, f"reintroduced in {marker!r}"


def test_notebook_agent_freezes_the_critic_for_the_actor_update():
    """The efficiency fix from the chapter's change plan, in both copies."""
    cell = find_cell("class DDPGAgent")
    assert "requires_grad" in cell
    assert "_set_critic_requires_grad(False)" in cell
    assert "_set_critic_requires_grad(True)" in cell


def test_notebook_soft_update_avoids_dot_data():
    cell = find_cell("class DDPGAgent")
    code = "\n".join(line.split("#")[0] for line in cell.splitlines())
    assert ".data.copy_" not in code
    assert "torch.no_grad()" in code


def test_notebook_training_loop_bootstraps_through_truncation():
    """Regression: storing ``float(terminated or truncated)`` zeroed the
    bootstrap at Pendulum's 200-step time limit, i.e. on every episode."""
    cell = find_cell("for ep in range(1, EPISODES + 1)")
    assert "float(terminated)" in cell
    assert "float(terminated or truncated)" not in cell
    assert "float(done)" not in cell


def test_notebook_training_loop_warms_up_with_random_actions():
    cell = find_cell("for ep in range(1, EPISODES + 1)")
    assert "WARMUP_STEPS" in cell
    assert "env.action_space.sample()" in cell


def test_notebook_agent_can_learn_on_pendulum(notebook_ns):
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    torch.manual_seed(0)
    np.random.seed(0)
    agent = notebook_ns["DDPGAgent"](
        state_dim=3, action_dim=1, max_action=2.0, batch_size=16)

    state, _ = env.reset(seed=0)
    for _ in range(80):
        action = agent.select_action(state)
        nxt, reward, terminated, truncated, _ = env.step(action)
        agent.store(state, action, reward, nxt, float(terminated))
        state = nxt if not (terminated or truncated) else env.reset()[0]

    before = [p.detach().clone() for p in agent.actor.parameters()]
    for _ in range(10):
        agent.train()
    after = list(agent.actor.parameters())

    assert any(not torch.equal(a, b) for a, b in zip(before, after))
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

    monkeypatch.setenv("CH4_NUM_EPISODES", "40")
    monkeypatch.chdir(CH4)

    namespace = {"__name__": "__notebook__"}
    for index, source in enumerate(code_cells()):
        try:
            exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)
        except Exception as exc:
            pytest.fail(
                f"Notebook cell {index} raised {type(exc).__name__}: {exc}\n"
                f"--- cell source ---\n{source[:600]}"
            )

    assert len(namespace["returns"]) == 40
    assert all(r < 0 for r in namespace["returns"])  # Pendulum rewards are negative
