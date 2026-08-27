import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.part_2_methods.ch05_ppo.ppo_agent import PPOAgent


def make_agent():
    torch.manual_seed(0)
    np.random.seed(0)
    return PPOAgent(
        state_dim=3,
        action_dim=1,
        max_action=2.0,
        batch_size=16,
        k_epochs=1,
    )


def test_action_mean_is_bounded_and_diagnostics_are_finite():
    agent = make_agent()
    state = np.zeros(3, dtype=np.float32)

    # Chapter 5 squashes only the distribution mean, so samples themselves are
    # unbounded; the environment clips them. Assert on the mean instead.
    with torch.no_grad():
        mean = agent.actor(torch.FloatTensor(state).unsqueeze(0)).mean
    assert torch.all(mean >= -2.0)
    assert torch.all(mean <= 2.0)

    for _ in range(100):
        action, logprob, value = agent.select_action(state)

        assert action.shape == (1,)
        assert np.all(np.isfinite(action))
        assert np.isfinite(logprob)
        assert np.isfinite(value)


def test_update_runs_on_a_small_rollout():
    agent = make_agent()
    state = np.zeros(3, dtype=np.float32)
    rollouts = []

    for _ in range(16):
        action, logprob, value = agent.select_action(state)
        rollouts.append((state, action, -1.0, state, 0.0, logprob, value))

    approx_kl, clip_fraction = agent.update(rollouts)

    assert np.isfinite(approx_kl)
    assert np.isfinite(clip_fraction)
    assert 0.0 <= clip_fraction <= 1.0


def test_empty_update_is_a_noop():
    assert make_agent().update([]) == (0.0, 0.0)


# --------------------------------------------------------------------------
# The ablation must stay in step with what chapter 5 teaches
# --------------------------------------------------------------------------

def _fake_result(episodes, value=-1000.0):
    """A RunResult shaped like a real run, cheap enough for a unit test."""
    from src.part_2_methods.ch05_ppo.train_pendulum import RunResult
    return RunResult([value] * episodes, [0.01, 0.02], [0.1, 0.2])


def test_ablation_default_variants_isolate_the_clip():
    """The default run contrasts clipping off against the published epsilon.

    Only ``eps_clip`` differs between the two, which is what lets the chapter
    attribute the whole gap to the clipped objective. A variant that changed
    anything else would make the figure unattributable.
    """
    from src.part_2_methods.ch05_ppo import ablation

    assert [row[0] for row in ablation.VARIANTS] == [
        "Clipping disabled", "PPO as published (eps = 0.2)",
    ]
    disabled, published = ablation.VARIANTS
    assert disabled[1] == ablation.NO_CLIP
    assert published[1] == 0.2

    styles = ablation.STYLES
    assert styles["Clipping disabled"]["ls"] == "--"
    assert styles["Clipping disabled"]["marker"] == "o"
    assert styles["PPO as published (eps = 0.2)"]["ls"] == "-."
    assert styles["PPO as published (eps = 0.2)"]["marker"] == "s"


def test_tight_clip_variant_is_opt_in():
    """The over-tight clip stays out of the default run, but stays runnable."""
    from src.part_2_methods.ch05_ppo import ablation

    assert ablation.TIGHT_VARIANT not in ablation.VARIANTS
    assert ablation.TIGHT_VARIANT == ("Over-tight clip (eps = 0.05)", 0.05)
    # Still styled, so --include-tight can plot it without a KeyError.
    assert "Over-tight clip (eps = 0.05)" in ablation.STYLES


def test_no_clip_never_binds():
    """``NO_CLIP`` has to be wide enough that torch.clamp is a no-op.

    If it were merely large-ish, the "clipping disabled" row would still be
    clipping occasionally and the ablation would understate the effect.
    """
    from src.part_2_methods.ch05_ppo import ablation

    ratios = torch.tensor([1e-6, 0.5, 1.0, 2.0, 1e3])
    clamped = torch.clamp(ratios, 1 - ablation.NO_CLIP, 1 + ablation.NO_CLIP)
    assert torch.equal(ratios, clamped)


def test_ablation_runs_every_variant_against_every_seed(monkeypatch):
    """Cheap orchestration check: the real run is minutes long, so the
    training call is stubbed and only the loop over variants x seeds runs."""
    from src.part_2_methods.ch05_ppo import ablation

    calls = []

    def fake_train(seed, episodes, verbose, **overrides):
        calls.append((seed, overrides["eps_clip"]))
        return _fake_result(episodes)

    monkeypatch.setattr(ablation, "train", fake_train)
    results = ablation.run(seeds=(0, 1), episodes=30, printer=lambda *a, **k: None)

    assert set(results) == {"Clipping disabled", "PPO as published (eps = 0.2)"}
    assert len(calls) == 4  # 2 variants x 2 seeds
    assert (0, ablation.NO_CLIP) in calls
    assert (0, 0.2) in calls


def test_sweep_covers_the_four_tuned_hyperparameters(monkeypatch):
    """Table 5.2 tunes epsilon, the learning rate, lambda and gamma."""
    from src.part_2_methods.ch05_ppo import ablation

    swept = [param for param, _, _, _ in ablation.SWEEPS]
    assert swept == ["eps_clip", "lr", "lam", "gamma"]

    for param, values, ticks, published in ablation.SWEEPS:
        assert len(values) == len(ticks), f"{param} ticks must label every value"
        assert published in values, f"{param} must sweep through its published value"


def test_sweep_reuses_the_shared_baseline_run(monkeypatch):
    """All four sweeps pass through the published configuration.

    Without the cache that run would be repeated once per sweep per seed,
    which is four times the CPU for identical numbers.
    """
    from src.part_2_methods.ch05_ppo import ablation

    calls = []

    def fake_train(seed, episodes, verbose, **overrides):
        calls.append((seed, tuple(sorted(overrides.items()))))
        return _fake_result(episodes)

    monkeypatch.setattr(ablation, "train", fake_train)
    ablation.run_sweep(seeds=(0,), episodes=10, printer=lambda *a, **k: None)

    assert len(calls) == len(set(calls)), "a configuration was trained twice"
    # 5 + 4 + 3 + 3 values, minus the 3 repeats of the shared baseline.
    assert len(calls) == 12


# --- seeding -------------------------------------------------------------

def draw_from_every_rng():
    """One draw from each generator a PPO run touches."""
    return (
        float(np.random.rand()),
        float(torch.rand(1)),
    )


def test_set_seed_makes_every_generator_reproducible():
    from src.part_2_methods.ch05_ppo.seeding import set_seed

    set_seed(42)
    first = draw_from_every_rng()
    set_seed(42)
    assert draw_from_every_rng() == first


def test_different_seeds_produce_different_draws():
    """Guards against a set_seed that freezes everything to a constant."""
    from src.part_2_methods.ch05_ppo.seeding import set_seed

    set_seed(42)
    a = draw_from_every_rng()
    set_seed(7)
    assert draw_from_every_rng() != a


def test_seeding_covers_network_initialization():
    """Weight init must be inside the seeded region, or two agents built from
    the same seed still differ."""
    from src.part_2_methods.ch05_ppo.seeding import set_seed

    def build():
        return PPOAgent(state_dim=3, action_dim=1, max_action=2.0)

    set_seed(42)
    first = [p.detach().clone() for p in build().actor.parameters()]
    set_seed(42)
    second = list(build().actor.parameters())

    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_set_seed_pins_the_thread_count():
    """The pin is part of seeding: thread count changes reduction order, and
    Chapter 5 prints an exact transcript."""
    from src.part_2_methods.ch05_ppo.seeding import set_seed

    torch.set_num_threads(2)
    set_seed(0)
    assert torch.get_num_threads() == 1


def test_episode_seed_is_the_convention_train_pendulum_uses():
    from src.part_2_methods.ch05_ppo.seeding import episode_seed

    assert episode_seed(42, 1) == 43
    assert episode_seed(0, 0) == 0


def test_episode_seed_streams_overlap_between_runs():
    """Documents a known limitation rather than asserting it is correct.

    ``seed + episode`` means run 0 and run 1 share every start state but one,
    shifted by an episode. The docstring explains why this is left alone; this
    test exists so that changing the stride is a deliberate act that updates a
    failing assertion, not a silent change to every number the chapter prints.
    """
    from src.part_2_methods.ch05_ppo.seeding import episode_seed

    run_a = {episode_seed(0, e) for e in range(1, 201)}
    run_b = {episode_seed(1, e) for e in range(1, 201)}
    assert len(run_a & run_b) == 199


def test_seeding_demo_runs_once_per_seed_and_reports_the_spread(monkeypatch):
    """The demo is what sources the chapter's spread threshold, so it has to
    train each seed exactly once and report max - min over the results."""
    from src.part_2_methods.ch05_ppo import seeding, train_pendulum

    scores = {0: -500.0, 1: -600.0, 2: -800.0}
    calls = []

    def fake_main(seed, episodes, verbose):
        calls.append(seed)
        return _fake_result(episodes, value=scores[seed])

    monkeypatch.setattr(train_pendulum, "main", fake_main)
    lines = []
    result = seeding._demo((0, 1, 2), episodes=30, printer=lines.append)

    assert calls == [0, 1, 2], "one run per seed, in order"
    assert result == [-500.0, -600.0, -800.0]
    assert "300.0" in "\n".join(lines), "the spread has to reach the reader"


# --- sample-efficiency figure ---------------------------------------------

def test_efficiency_styles_cover_every_algorithm():
    """A label with no STYLES entry raises KeyError inside plot(), long after
    the runs that produced it have been paid for."""
    from src.part_2_methods.ch05_ppo import plot_efficiency

    for label, _ in plot_efficiency.ALGORITHMS:
        assert label in plot_efficiency.STYLES


def test_efficiency_runs_every_algorithm_against_every_seed():
    """``run`` binds ALGORITHMS as a default argument, so a caller overriding
    it passes it in rather than patching the module attribute."""
    from src.part_2_methods.ch05_ppo import plot_efficiency

    calls = []

    def fake(name):
        def runner(seed, episodes):
            calls.append((name, seed))
            return [-1000.0] * episodes
        return runner

    results = plot_efficiency.run(
        seeds=(0, 1), episodes=10, printer=lambda *a, **k: None,
        algorithms=(("DDPG (chapter 4)", fake("ddpg")),
                    ("PPO (chapter 5)", fake("ppo")),
                    ("SAC (chapter 6)", fake("sac"))))

    assert len(calls) == 6, "three algorithms x two seeds"
    assert set(results) == {"DDPG (chapter 4)", "PPO (chapter 5)",
                            "SAC (chapter 6)"}
    assert all(len(c) == 10 for curves in results.values() for c in curves)


def test_sac_main_returns_its_curve():
    """Regression guard. ``main`` used to end on a print and return None, so
    every caller that wanted the curve got a TypeError instead."""
    from src.part_2_methods.ch06_sac.train_pendulum import main

    returns = main(seed=0, total_steps=400, verbose=False)

    assert isinstance(returns, list)
    assert len(returns) == 2, "400 steps of 200-step episodes"
    assert all(isinstance(r, float) for r in returns)


def test_sac_seed_reaches_the_environment():
    """Regression guard. The environment was reset with the module-level SEED
    rather than the argument, so every seed replayed one episode stream."""
    from src.part_2_methods.ch06_sac.train_pendulum import main

    assert main(seed=0, total_steps=400, verbose=False) != \
        main(seed=1, total_steps=400, verbose=False)
