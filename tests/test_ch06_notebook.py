"""Tests for the Chapter 6 notebook.

The notebook re-implements SAC inline so readers can follow the derivation cell
by cell. That duplication is a teaching choice, but it is also the mechanism by
which a notebook drifts away from the modules it mirrors -- chapter 5's had
drifted in both directions at once before anyone tested it.

These tests cover both risks:

  * parity    -- the notebook's inline classes still behave like the modules
  * execution -- the notebook actually runs top to bottom without raising

The notebook deliberately does *not* mirror everything. ``SACAgent`` there is
Algorithm 1 and nothing else; the module adds the ``auto_alpha``,
``init_alpha`` and ``reward_scale`` switches that ``ablation.py`` needs, and
that asymmetry is intentional rather than drift.

The execution test is marked ``slow`` because it runs a real training loop.
Run everything with ``make test-all``; the default ``make test`` skips it.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

CH6 = (
    Path(__file__).resolve().parents[1]
    / "src" / "part_2_methods" / "ch06_sac"
)
NOTEBOOK = CH6 / "Chapter6_SAC.ipynb"


def code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "code"]


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
    for marker in ("import gymnasium", "class Actor", "class Critic",
                   "class ReplayBuffer", "class SACAgent"):
        exec(compile(find_cell(marker), "<notebook>", "exec"), namespace)
    return namespace


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

def test_notebook_is_valid_json_and_every_cell_has_an_id():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] >= 5
    assert nb["cells"]
    # nbformat 4.5 requires cell ids; chapter 4 shipped without them once.
    assert all("id" in cell for cell in nb["cells"])


def test_notebook_defines_the_classes_the_chapter_describes(notebook_ns):
    for name in ("Actor", "Critic", "ReplayBuffer", "SACAgent"):
        assert name in notebook_ns, f"notebook no longer defines {name}"


def test_training_cell_budget_is_overridable():
    """Guards the hook the execution test depends on."""
    cell = find_cell("TOTAL_STEPS")
    assert "CH6_TOTAL_STEPS" in cell
    assert "CH6_WARMUP_STEPS" in cell


def test_colab_install_pins_match_the_requirements_files():
    """The Colab cell is the only install path a reader on Colab ever sees, so
    it must not drift from the repo's own bounds.

    Regression: it pinned ``gymnasium[classic-control]>=0.28.1`` against the
    repo's ``>=1.0,<2.0``.
    """
    install = find_cell("pip install")
    assert '"gymnasium[classic-control]>=1.0,<2.0"' in install
    assert '"torch>=2.0.0"' in install


def test_setup_cell_seeds_the_buffers_sampler_and_pins_the_threads():
    """SAC draws its minibatch through ``random.sample``. A notebook that
    seeds only numpy and torch prints a different transcript every run while
    looking fully seeded."""
    setup = executable(find_cell("SEED = 42"))
    assert "random.seed(SEED)" in setup
    assert "np.random.seed(SEED)" in setup
    assert "torch.manual_seed(SEED)" in setup
    assert "torch.set_num_threads(1)" in setup


def test_training_cell_bootstraps_through_truncation():
    """Pendulum-v1 never terminates, only truncates, so storing
    ``float(terminated or truncated)`` would zero the bootstrap at the end of
    every single episode."""
    code = executable(find_cell("TOTAL_STEPS"))
    assert "float(terminated)" in code
    assert "float(terminated or truncated)" not in code
    assert "float(done)" not in code


def test_training_cell_seeds_the_action_space():
    """The warmup drives the environment entirely from
    ``action_space.sample()``."""
    assert "env.action_space.seed(SEED)" in executable(find_cell("TOTAL_STEPS"))


# --------------------------------------------------------------------------
# Parity with the module implementations
# --------------------------------------------------------------------------

def test_notebook_actor_matches_module_architecture(notebook_ns):
    from src.part_2_methods.ch06_sac import Actor as ModuleActor
    from src.part_2_methods.ch06_sac import Critic as ModuleCritic

    nb_actor = notebook_ns["Actor"](3, 1, max_action=2.0)
    mod_actor = ModuleActor(3, 1, max_action=2.0)
    nb_critic, mod_critic = notebook_ns["Critic"](3, 1), ModuleCritic(3, 1)

    def shapes(net):
        return [(m.in_features, m.out_features)
                for m in net.modules() if isinstance(m, torch.nn.Linear)]

    assert shapes(nb_actor) == shapes(mod_actor)
    assert shapes(nb_critic) == shapes(mod_critic)


def test_notebook_actor_matches_module_log_probabilities(notebook_ns):
    """Same weights and same noise in, same action and log-probability out.

    This is the test that would have caught the correction drifting: the
    module used ``log(max_action * (1 - y^2) + 1e-6)`` while listing 6.1 uses
    the softplus form, and the two differ by a constant per action dimension.
    """
    from src.part_2_methods.ch06_sac import Actor as ModuleActor

    nb_actor = notebook_ns["Actor"](3, 1, max_action=2.0)
    mod_actor = ModuleActor(3, 1, max_action=2.0)
    mod_actor.load_state_dict(nb_actor.state_dict())

    states = torch.randn(16, 3)
    torch.manual_seed(11)
    nb_action, nb_logp = nb_actor(states)
    torch.manual_seed(11)
    mod_action, mod_logp = mod_actor(states)

    assert torch.allclose(nb_action, mod_action)
    assert torch.allclose(nb_logp, mod_logp)


def test_notebook_actor_applies_the_tanh_correction(notebook_ns):
    """The bug the chapter calls the most common one in SAC."""
    code = executable(find_cell("class Actor"))
    assert "F.softplus(-2 * u)" in code
    assert "rsample()" in code, "sampling must be reparameterized"

    _, log_prob = notebook_ns["Actor"](3, 1, max_action=2.0)(torch.randn(8, 3))
    assert torch.all(torch.isfinite(log_prob))


def test_notebook_critic_returns_two_independent_q_values(notebook_ns):
    critic = notebook_ns["Critic"](3, 1)
    q1, q2 = critic(torch.randn(16, 3), torch.randn(16, 1))
    assert q1.shape == q2.shape == (16, 1)
    assert not torch.allclose(q1, q2)


def test_notebook_agent_takes_the_min_and_learns_a_temperature(notebook_ns):
    code = executable(find_cell("class SACAgent"))
    assert "torch.min(q1_t, q2_t)" in code
    assert "log_alpha" in code and "target_entropy" in code
    # No target actor: the stochastic policy smooths the target on its own.
    assert "actor_target" not in code


def test_notebook_agent_update_moves_all_three_optimizers(notebook_ns):
    agent = notebook_ns["SACAgent"](3, 1, 2.0, batch_size=8)
    for _ in range(32):
        s = np.random.randn(3).astype(np.float32)
        agent.store(s, np.array([0.1], dtype=np.float32), -1.0, s, 0.0)

    before_actor = [p.detach().clone() for p in agent.actor.parameters()]
    before_critic = [p.detach().clone() for p in agent.critic.parameters()]
    before_alpha = agent.log_alpha.detach().clone()

    critic_loss, actor_loss, alpha = agent.train()

    assert np.isfinite(critic_loss) and np.isfinite(actor_loss) and alpha > 0
    assert any(not torch.equal(a, b)
               for a, b in zip(before_actor, agent.actor.parameters()))
    assert any(not torch.equal(a, b)
               for a, b in zip(before_critic, agent.critic.parameters()))
    assert not torch.equal(before_alpha, agent.log_alpha)


def test_notebook_agent_default_target_entropy_is_minus_action_dim(notebook_ns):
    assert notebook_ns["SACAgent"](3, 1, 2.0).target_entropy == -1.0
    assert notebook_ns["SACAgent"](3, 6, 1.0).target_entropy == -6.0


def test_evaluation_cell_uses_the_mean_action(notebook_ns):
    """The chapter's Tip: score a trained policy on its mean action, not a
    sample, because it is trained against the entropy-augmented objective and
    scored against the plain reward."""
    assert "deterministic=True" in executable(find_cell("eval_env"))


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

    monkeypatch.setenv("CH6_WARMUP_STEPS", "400")
    monkeypatch.setenv("CH6_TOTAL_STEPS", "1200")
    monkeypatch.chdir(CH6)

    namespace = {"__name__": "__notebook__"}
    for index, source in enumerate(code_cells()):
        try:
            exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)
        except Exception as exc:
            pytest.fail(
                f"Notebook cell {index} raised {type(exc).__name__}: {exc}\n"
                f"--- cell source ---\n{source[:600]}"
            )

    assert len(namespace["ep_returns"]) == 6   # 1200 steps of 200-step episodes
    assert all(r < 0 for r in namespace["ep_returns"])  # Pendulum is negative
    assert len(namespace["ep_alphas"]) == len(namespace["ep_returns"])
    assert namespace["eval_returns"], "the evaluation cell produced nothing"
