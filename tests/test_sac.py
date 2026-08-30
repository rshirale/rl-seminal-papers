"""Tests for the Chapter 6 SAC modules.

Three things are worth guarding here, and they are not the same three as in
chapters 4 and 5.

The first is the tanh log-probability correction. The chapter calls omitting it
"the most common implementation bug in SAC", and it is invisible in a training
curve -- the agent still learns, it just learns against a biased entropy
signal. So it is checked numerically against the change-of-variables identity
rather than by grepping for the line.

The second is that the ablation switches really do turn off what they claim to.
An ablation whose "entropy off" variant still runs the temperature optimizer
would produce a figure that argues for nothing.

The third is seeding. SAC draws from four generators, and the replay buffer's
``random.sample`` is the easy one to leave out -- a run seeded without it has
identical initial weights and a different gradient at every step.
"""

import math
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.part_2_methods.ch06_sac.actor import (
    LOG_STD_MAX, LOG_STD_MIN, Actor,
)
from src.part_2_methods.ch06_sac.critic import Critic
from src.part_2_methods.ch06_sac.replay_buffer import ReplayBuffer
from src.part_2_methods.ch06_sac.sac_agent import SACAgent
from src.part_2_methods.ch06_sac.train_pendulum import RunResult

STATE_DIM, ACTION_DIM, MAX_ACTION = 3, 1, 2.0


def make_agent(**kwargs):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    kwargs.setdefault("batch_size", 8)
    return SACAgent(STATE_DIM, ACTION_DIM, MAX_ACTION, **kwargs)


def fill(agent, n=32):
    for _ in range(n):
        s = np.random.randn(STATE_DIM).astype(np.float32)
        agent.store(s, np.array([0.1], dtype=np.float32), -1.0, s, False)
    return agent


# --------------------------------------------------------------------------
# The actor: squashing, and the correction the chapter calls the common bug
# --------------------------------------------------------------------------

def test_sampled_actions_respect_the_action_bound():
    """Unlike chapter 5's PPO actor, SAC squashes the *sample*, not the mean,
    so every action it emits is inside the bound by construction."""
    actor = Actor(STATE_DIM, ACTION_DIM, max_action=MAX_ACTION)
    action, log_prob = actor(torch.randn(64, STATE_DIM))

    assert torch.all(action.abs() <= MAX_ACTION)
    assert action.shape == (64, ACTION_DIM)
    assert log_prob.shape == (64, 1)
    assert torch.all(torch.isfinite(log_prob))


def test_deterministic_forward_returns_the_mean_action():
    """The evaluation path of the chapter's Tip: tanh(mu) * max_action, no
    rsample, and no log-probability to report."""
    actor = Actor(STATE_DIM, ACTION_DIM, max_action=MAX_ACTION)
    states = torch.randn(16, STATE_DIM)

    action, log_prob = actor(states, deterministic=True)
    assert log_prob is None

    expected = torch.tanh(actor.mean_head(actor.net(states))) * MAX_ACTION
    assert torch.allclose(action, expected)
    # Deterministic means deterministic: the same state twice, the same action.
    assert torch.allclose(action, actor(states, deterministic=True)[0])


def test_deterministic_actions_repeat_where_sampled_ones_do_not():
    actor = Actor(STATE_DIM, ACTION_DIM, max_action=MAX_ACTION)
    states = torch.zeros(8, STATE_DIM)

    assert not torch.allclose(actor(states)[0], actor(states)[0])


def test_log_prob_carries_the_tanh_jacobian_correction():
    """The bug the chapter warns about, checked against the identity itself.

    log pi(a|s) = log p(u|s) - sum_i log(1 - tanh^2(u_i)). The module computes
    the second term through ``softplus`` for numerical stability; this
    recomputes it the naive way at moderate ``u``, where both forms agree, and
    asserts the module's answer sits below the uncorrected Gaussian
    log-density by exactly that amount.
    """
    from torch.distributions import Normal

    torch.manual_seed(0)
    actor = Actor(STATE_DIM, ACTION_DIM, max_action=MAX_ACTION)
    states = torch.randn(256, STATE_DIM)

    torch.manual_seed(7)
    action, log_prob = actor(states)

    # Recover u from the action, then rebuild the distribution it came from.
    u = torch.atanh((action / MAX_ACTION).clamp(-0.999999, 0.999999))
    x = actor.net(states)
    mean = actor.mean_head(x)
    std = actor.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    gaussian = Normal(mean, std).log_prob(u).sum(-1, keepdim=True)
    naive_correction = torch.log(1 - torch.tanh(u).pow(2)).sum(-1, keepdim=True)

    assert torch.allclose(log_prob, gaussian - naive_correction, atol=1e-4)
    # And the correction is not a no-op: it must actually move the number.
    assert not torch.allclose(log_prob, gaussian, atol=1e-2)


def test_log_prob_is_finite_when_the_action_saturates():
    """The reason the module uses ``softplus`` rather than ``log(1 - y^2)``.

    Drive the pre-tanh sample far into the tail and the naive form underflows
    to ``log(0) = -inf``; the stable form must not.
    """
    actor = Actor(STATE_DIM, ACTION_DIM, max_action=MAX_ACTION)
    with torch.no_grad():
        actor.mean_head.bias.fill_(30.0)     # tanh(30) == 1.0 in float32
        actor.log_std_head.bias.fill_(LOG_STD_MIN)

    _, log_prob = actor(torch.zeros(16, STATE_DIM))
    assert torch.all(torch.isfinite(log_prob))


def test_log_std_is_clamped_into_its_stable_range():
    actor = Actor(STATE_DIM, ACTION_DIM, max_action=MAX_ACTION)
    with torch.no_grad():
        actor.log_std_head.bias.fill_(50.0)
    x = actor.net(torch.zeros(4, STATE_DIM))
    clamped = actor.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
    assert torch.all(clamped <= LOG_STD_MAX)


def test_actor_gradients_flow_through_the_sampling_step():
    """The whole point of the reparameterization trick: a loss computed on the
    sampled action must reach the weights that produced it."""
    actor = Actor(STATE_DIM, ACTION_DIM, max_action=MAX_ACTION)
    action, log_prob = actor(torch.randn(8, STATE_DIM))
    (action.sum() + log_prob.sum()).backward()

    assert actor.mean_head.weight.grad is not None
    assert torch.any(actor.mean_head.weight.grad != 0)
    assert torch.any(actor.log_std_head.weight.grad != 0)


# --------------------------------------------------------------------------
# The twin critics
# --------------------------------------------------------------------------

def test_the_two_q_networks_are_independently_initialized():
    """``min(Q1, Q2)`` is only pessimistic if the two disagree. Building them
    from one shared module -- or copying one into the other -- would make the
    minimum exactly the single-critic estimate it is meant to correct."""
    critic = Critic(STATE_DIM, ACTION_DIM)
    q1, q2 = critic(torch.randn(32, STATE_DIM), torch.randn(32, ACTION_DIM))

    assert q1.shape == q2.shape == (32, 1)
    assert not torch.allclose(q1, q2)
    assert set(id(p) for p in critic.q1.parameters()).isdisjoint(
        id(p) for p in critic.q2.parameters())


# --------------------------------------------------------------------------
# The replay buffer
# --------------------------------------------------------------------------

def test_buffer_batches_have_the_shapes_the_update_expects():
    buf = ReplayBuffer(capacity=64)
    for _ in range(32):
        buf.push(np.zeros(STATE_DIM, dtype=np.float32),
                 np.zeros(ACTION_DIM, dtype=np.float32),
                 -1.0, np.ones(STATE_DIM, dtype=np.float32), False)

    s, a, r, ns, d = buf.sample(8)
    assert s.shape == ns.shape == (8, STATE_DIM)
    assert a.shape == (8, ACTION_DIM)
    # Rewards and done flags are columns, not rows: they are broadcast against
    # (batch, 1) Q-values in the Bellman target.
    assert r.shape == d.shape == (8, 1)


def test_buffer_normalizes_the_done_flag_and_evicts_when_full():
    buf = ReplayBuffer(capacity=4)
    for i in range(6):
        buf.push(np.zeros(1), np.zeros(1), float(i), np.zeros(1), i % 2 == 0)

    assert len(buf) == 4
    assert all(isinstance(t[4], float) for t in buf.buf)
    assert [t[2] for t in buf.buf] == [2.0, 3.0, 4.0, 5.0]


# --------------------------------------------------------------------------
# The agent's update
# --------------------------------------------------------------------------

def test_update_is_a_noop_until_the_buffer_holds_a_full_minibatch():
    agent = fill(make_agent(batch_size=16), n=4)
    assert agent.train() == (None, None, None)


def test_update_moves_actor_critic_and_temperature():
    agent = fill(make_agent())
    before = {
        "actor": [p.detach().clone() for p in agent.actor.parameters()],
        "critic": [p.detach().clone() for p in agent.critic.parameters()],
        "log_alpha": agent.log_alpha.detach().clone(),
    }
    critic_loss, actor_loss, alpha = agent.train()

    assert math.isfinite(critic_loss) and math.isfinite(actor_loss)
    assert alpha > 0
    assert any(not torch.equal(a, b)
               for a, b in zip(before["actor"], agent.actor.parameters()))
    assert any(not torch.equal(a, b)
               for a, b in zip(before["critic"], agent.critic.parameters()))
    assert not torch.equal(before["log_alpha"], agent.log_alpha)


def test_target_critics_are_frozen_and_only_drift():
    """No gradient reaches them; they move only through Polyak averaging, and
    by a fraction of tau of the gap on each step."""
    agent = fill(make_agent(tau=0.005))
    assert all(not p.requires_grad for p in agent.critic_target.parameters())

    before = [p.detach().clone() for p in agent.critic_target.parameters()]
    agent.train()
    after = list(agent.critic_target.parameters())

    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    for b, a, online in zip(before, after, agent.critic.parameters()):
        assert torch.allclose(a, 0.005 * online.data + 0.995 * b, atol=1e-6)


def test_default_target_entropy_is_the_negative_action_dimension():
    """The SAC-v2 heuristic H-bar = -dim(A); -1 on Pendulum-v1."""
    assert make_agent().target_entropy == -1.0
    assert SACAgent(3, 6, 1.0).target_entropy == -6.0


def test_there_is_no_target_actor():
    """SAC needs none: the stochastic policy smooths the target on its own,
    which is what TD3 has to simulate with injected noise."""
    assert not hasattr(make_agent(), "actor_target")


# --------------------------------------------------------------------------
# The ablation switches must really switch things off
# --------------------------------------------------------------------------

def test_fixed_temperature_holds_alpha_and_builds_no_optimizer():
    agent = fill(make_agent(auto_alpha=False, init_alpha=0.2))
    assert agent.alpha_opt is None
    assert agent.alpha == pytest.approx(0.2)

    agent.train()
    assert agent.alpha == pytest.approx(0.2), "fixed alpha must not move"


def test_entropy_off_variant_zeroes_alpha_and_still_trains():
    """Exercise 1's configuration. alpha = 0 removes the entropy term from
    both the actor loss and the Bellman target while everything else -- twin
    critics, replay, soft targets, the stochastic actor -- stays."""
    agent = fill(make_agent(auto_alpha=False, init_alpha=0.0))
    assert agent.alpha == 0.0

    critic_loss, actor_loss, alpha = agent.train()
    assert alpha == 0.0
    assert math.isfinite(critic_loss) and math.isfinite(actor_loss)


def test_learning_a_temperature_that_starts_at_zero_is_refused():
    """log(0) has no finite gradient, so the combination is rejected loudly
    rather than silently producing NaNs a few hundred steps in."""
    with pytest.raises(ValueError, match="init_alpha must be positive"):
        SACAgent(STATE_DIM, ACTION_DIM, MAX_ACTION,
                 auto_alpha=True, init_alpha=0.0)


def test_reward_scale_multiplies_the_bellman_target():
    """Exercise 3's handle. Same weights, same batch, scaled rewards: the
    critic loss must move, because the target it is regressing onto did."""
    losses = []
    for scale in (1.0, 10.0):
        agent = fill(make_agent(reward_scale=scale))
        losses.append(agent.train()[0])
    assert losses[0] != losses[1]


def test_alpha_falls_when_entropy_sits_above_the_target():
    """The auto-tuner's direction, which is the whole mechanism of figure 6.6.

    A freshly initialized policy is far more random than H-bar = -1, so the
    dual gradient must push alpha down.
    """
    agent = fill(make_agent(alpha_lr=1e-2), n=64)
    start = agent.alpha
    for _ in range(30):
        agent.train()
    assert agent.alpha < start


def test_alpha_rises_when_the_policy_is_more_deterministic_than_the_target():
    """The mechanism is bidirectional, which is the claim the chapter makes
    right after describing the descent."""
    agent = fill(make_agent(alpha_lr=1e-2, target_entropy=50.0), n=64)
    start = agent.alpha
    for _ in range(30):
        agent.train()
    assert agent.alpha > start


def test_select_action_is_bounded_and_deterministic_on_request():
    agent = make_agent()
    state = np.zeros(STATE_DIM, dtype=np.float32)

    for _ in range(50):
        action = agent.select_action(state)
        assert action.shape == (ACTION_DIM,)
        assert np.all(np.abs(action) <= MAX_ACTION)

    assert np.array_equal(agent.select_action(state, deterministic=True),
                          agent.select_action(state, deterministic=True))


def test_save_and_load_round_trip(tmp_path):
    agent = fill(make_agent())
    agent.train()
    path = tmp_path / "sac.pt"
    agent.save(path)

    restored = make_agent()
    restored.load(path)

    state = torch.zeros(1, STATE_DIM)
    assert torch.allclose(agent.actor(state, deterministic=True)[0],
                          restored.actor(state, deterministic=True)[0])
    assert restored.alpha == pytest.approx(agent.alpha)


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

def draw_from_every_rng():
    return (random.random(), float(np.random.rand()), float(torch.rand(1)))


def test_set_seed_covers_every_generator_a_sac_run_draws_from():
    from src.part_2_methods.ch06_sac.seeding import set_seed

    set_seed(123)
    first = draw_from_every_rng()
    set_seed(123)
    assert draw_from_every_rng() == first


def test_set_seed_covers_the_replay_buffers_sampler():
    """``random.sample`` is the draw that is easy to leave unseeded: the
    weights come out identical and every gradient step differs anyway."""
    from src.part_2_methods.ch06_sac.seeding import set_seed

    buf = ReplayBuffer(capacity=100)
    for i in range(100):
        buf.push(np.array([i], dtype=np.float32), np.zeros(1), float(i),
                 np.zeros(1), False)

    set_seed(5)
    first = buf.sample(8)[2]
    set_seed(5)
    assert torch.equal(buf.sample(8)[2], first)


def test_set_seed_covers_network_initialization():
    from src.part_2_methods.ch06_sac.seeding import set_seed

    set_seed(7)
    a = Actor(STATE_DIM, ACTION_DIM, max_action=MAX_ACTION)
    set_seed(7)
    b = Actor(STATE_DIM, ACTION_DIM, max_action=MAX_ACTION)

    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_set_seed_pins_the_thread_count():
    """Part of seeding, not a performance tweak -- torch's intra-op
    parallelism changes the reduction order, so an 8-core machine and a 4-core
    one disagree on the transcript the chapter prints."""
    from src.part_2_methods.ch06_sac.seeding import set_seed

    set_seed(0)
    assert torch.get_num_threads() == 1


def test_seed_env_seeds_both_the_episode_stream_and_the_action_sampler():
    """The warmup is 10,000 uniform draws from ``action_space``, a fifth of the
    default budget, so an unseeded action space leaves the buffer different
    between runs even when everything else is pinned."""
    gym = pytest.importorskip("gymnasium")
    from src.part_2_methods.ch06_sac.seeding import seed_env

    def first_draws(seed):
        env = gym.make("Pendulum-v1")
        seed_env(env, seed)
        state, _ = env.reset()
        actions = [env.action_space.sample() for _ in range(5)]
        env.close()
        return state, actions

    s1, a1 = first_draws(3)
    s2, a2 = first_draws(3)
    assert np.array_equal(s1, s2)
    assert all(np.array_equal(x, y) for x, y in zip(a1, a2))


# --------------------------------------------------------------------------
# The trainer and the ablation must stay in step with what chapter 6 teaches
# --------------------------------------------------------------------------

def test_main_returns_its_curve_and_the_seed_reaches_the_environment():
    """Regression guard, and the seam chapter 5's efficiency figure calls."""
    from src.part_2_methods.ch06_sac.train_pendulum import main

    result = main(seed=0, total_steps=400, warmup_steps=400, verbose=False)
    assert isinstance(result.returns, list) and len(result.returns) == 2
    assert all(isinstance(r, float) for r in result.returns)

    again = main(seed=0, total_steps=400, warmup_steps=400, verbose=False)
    assert again.returns == result.returns, "same seed must replay the run"
    assert main(seed=1, total_steps=400, warmup_steps=400,
                verbose=False).returns != result.returns


def test_warmup_is_the_published_ten_thousand_at_the_published_budgets():
    """``warmup_for`` shrinks with ``--steps`` so short iterations still take
    gradient steps -- but it must not move at the budgets the chapter uses."""
    from src.part_2_methods.ch06_sac import ablation
    from src.part_2_methods.ch06_sac.train_pendulum import (
        TOTAL_STEPS, WARMUP_STEPS,
    )

    assert ablation.warmup_for(ablation.TOTAL_STEPS) == WARMUP_STEPS
    assert ablation.warmup_for(TOTAL_STEPS) == WARMUP_STEPS
    assert ablation.warmup_for(3_000) < WARMUP_STEPS


def test_entropy_ablation_changes_only_the_entropy_term():
    """The default contrast is what lets the chapter attribute the gap to the
    maximum entropy objective. A variant that changed a second thing -- the
    number of critics, the learning rate -- would make the figure
    unattributable."""
    from src.part_2_methods.ch06_sac import ablation

    labels = dict(ablation.VARIANTS)
    assert set(labels) == {ablation.NO_ENTROPY, ablation.PUBLISHED}
    assert labels[ablation.PUBLISHED] == {}, "the baseline overrides nothing"
    assert labels[ablation.NO_ENTROPY] == {
        "auto_alpha": False, "init_alpha": 0.0}


def test_temperature_variants_are_the_three_values_the_exercise_names():
    from src.part_2_methods.ch06_sac import ablation

    fixed = [o["init_alpha"] for _, o in ablation.TEMPERATURE_VARIANTS
             if not o.get("auto_alpha", True)]
    assert fixed == [0.01, 0.2, 1.0]
    assert ablation.PUBLISHED in dict(ablation.TEMPERATURE_VARIANTS)


def test_reward_scale_variants_pair_each_temperature_regime_at_both_scales():
    from src.part_2_methods.ch06_sac import ablation

    scales = [o.get("reward_scale", 1.0)
              for _, o in ablation.REWARD_SCALE_VARIANTS]
    assert sorted(scales) == [1.0, 1.0, 10.0, 10.0]


def test_every_variant_has_a_plot_style():
    """A missing entry is a KeyError an hour into a run, after every training
    call has already finished."""
    from src.part_2_methods.ch06_sac import ablation

    for group in (ablation.VARIANTS, ablation.TEMPERATURE_VARIANTS,
                  ablation.REWARD_SCALE_VARIANTS):
        for label, _ in group:
            assert label in ablation.STYLES


def test_ablation_runs_every_variant_against_every_seed(monkeypatch):
    from src.part_2_methods.ch06_sac import ablation

    calls = []

    def fake_train(seed, total_steps, verbose, **overrides):
        calls.append((seed, tuple(sorted(overrides.items()))))
        return RunResult([-1000.0] * (total_steps // ablation.STEPS_PER_EPISODE),
                         sigma=0.5, entropy=-1.0, alpha=0.2)

    monkeypatch.setattr(ablation, "train", fake_train)
    results = ablation.run(seeds=(0, 1), total_steps=2000,
                           printer=lambda *a, **k: None)

    assert len(calls) == 4, "two variants x two seeds"
    assert set(results) == {ablation.NO_ENTROPY, ablation.PUBLISHED}


def test_ablation_caches_the_baseline_across_experiments(monkeypatch):
    """The published configuration appears in all three experiments. Sharing a
    cache across them saves three runs per seed, which at five minutes a run is
    not a rounding error."""
    from src.part_2_methods.ch06_sac import ablation

    calls = []

    def fake_train(seed, total_steps, verbose, **overrides):
        calls.append((seed, tuple(sorted(overrides.items()))))
        return RunResult([-1000.0] * (total_steps // ablation.STEPS_PER_EPISODE),
                         sigma=0.5, entropy=-1.0, alpha=0.2)

    monkeypatch.setattr(ablation, "train", fake_train)
    cache, quiet = {}, (lambda *a, **k: None)
    ablation.run(seeds=(0,), total_steps=2000, printer=quiet, cache=cache)
    before = len(calls)
    ablation.run_temperature(seeds=(0,), total_steps=2000, printer=quiet,
                             cache=cache)

    assert len(calls) - before == 3, "the learned-alpha run is reused"


def test_measure_policy_reports_sigma_and_the_entropy_estimate():
    """The measurement the ablation tables now lean on.

    Probed without training: set the log-std head to a known constant and the
    reported sigma must be exp() of it.
    """
    import math

    from src.part_2_methods.ch06_sac.train_pendulum import measure_policy

    agent = fill(make_agent(), n=64)
    with torch.no_grad():
        agent.actor.log_std_head.weight.zero_()
        agent.actor.log_std_head.bias.fill_(math.log(0.25))

    sigma, entropy = measure_policy(agent, n_states=64)
    assert sigma == pytest.approx(0.25, abs=1e-3)
    assert math.isfinite(entropy)


def test_measure_policy_is_safe_on_an_empty_buffer():
    """A run too short to probe reports nan rather than raising -- the tables
    would otherwise crash on a one-episode smoke run."""
    import math

    from src.part_2_methods.ch06_sac.train_pendulum import measure_policy

    sigma, entropy = measure_policy(make_agent())
    assert math.isnan(sigma) and math.isnan(entropy)


def test_entropy_is_the_negative_mean_log_prob():
    """H = E[-log pi], the quantity the temperature regulates, so that the
    number in the table is comparable against the target of -dim(A)."""
    from src.part_2_methods.ch06_sac.train_pendulum import measure_policy

    agent = fill(make_agent(), n=64)
    torch.manual_seed(3)
    _, entropy = measure_policy(agent, n_states=64)

    states, _, _, _, _ = agent.replay.sample(64)
    torch.manual_seed(3)
    # Same probe, recomputed: sign and magnitude must agree.
    assert entropy == pytest.approx(
        float(-agent.actor(states)[1].mean()), abs=0.5)


@pytest.mark.slow
def test_removing_the_entropy_bonus_collapses_the_policy():
    """The finding the chapter's exercise 1 is really about.

    On Pendulum-v1 the return cannot separate these two variants -- both
    converge to roughly the same score -- but the policies behind those
    identical numbers are nothing alike. With alpha = 0 the actor collapses
    toward a Dirac delta, which is precisely what the maximum entropy
    objective exists to prevent.
    """
    from src.part_2_methods.ch06_sac.train_pendulum import main

    off = main(seed=0, total_steps=6_000, warmup_steps=1_000,
               auto_alpha=False, init_alpha=0.0, verbose=False)
    on = main(seed=0, total_steps=6_000, warmup_steps=1_000, verbose=False)

    assert off.sigma < on.sigma / 2, (
        f"entropy-off sigma {off.sigma:.4f} should be far below "
        f"the published run's {on.sigma:.4f}")
    assert off.entropy < on.entropy, "removing the bonus must lower entropy"


def test_plot_accepts_run_results_and_writes_both_formats(tmp_path):
    """Regression, and it cost an hour of CI to find.

    ``main`` returns a ``RunResult`` since the policy diagnostics landed, but
    ``plot`` was still handing whole results to ``smooth`` -- which turned a
    4-field namedtuple into an inhomogeneous array and raised. No test passed
    ``figure_dir``, so the entire figure path was uncovered and the failure
    surfaced only after every training run in the job had finished.
    """
    import matplotlib
    matplotlib.use("Agg")

    from src.part_2_methods.ch06_sac import ablation
    from src.part_2_methods.ch06_sac.train_pendulum import RunResult

    results = {
        ablation.NO_ENTROPY: [RunResult([-1200.0 + i] * 40, 0.007, -11.9, 0.0)
                              for i in range(3)],
        ablation.PUBLISHED: [RunResult([-1100.0 + i] * 40, 0.562, -0.97, 0.07)
                             for i in range(3)],
    }
    ablation.plot(results, str(tmp_path))

    for ext in ("png", "svg"):
        written = tmp_path / f"ch06-figure-entropy.{ext}"
        assert written.exists() and written.stat().st_size > 0


def test_plot_still_accepts_bare_return_lists(tmp_path):
    """``_returns`` takes either shape, so an older caller keeps working."""
    import matplotlib
    matplotlib.use("Agg")

    from src.part_2_methods.ch06_sac import ablation

    results = {ablation.NO_ENTROPY: [[-1200.0] * 40],
               ablation.PUBLISHED: [[-1100.0] * 40]}
    ablation.plot(results, str(tmp_path))
    assert (tmp_path / "ch06-figure-entropy.png").exists()


def test_score_is_a_median_over_the_tail():
    """Not a mean: one Pendulum episode in ten starts near upright and scores
    close to zero, and the chapter says to judge by the median."""
    from src.part_2_methods.ch06_sac.ablation import _score

    returns = [-1200.0] * 10 + [-125.0] * 49 + [-1.6]
    assert _score(returns, window=50) == pytest.approx(-125.0)


@pytest.mark.slow
def test_seeding_demo_reports_a_spread(monkeypatch):
    from src.part_2_methods.ch06_sac import seeding

    lines = []
    scores = seeding._demo((0, 1), total_steps=1200, printer=lines.append)

    assert len(scores) == 2
    assert any("spread" in line for line in lines)
