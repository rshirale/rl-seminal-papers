"""Chapter 4 (DDPG) module tests.

Covers the actor, critic, both noise processes, the replay buffer, and the
agent. Several tests are explicit regressions for bugs that were live in this
chapter; those carry a "Regression:" note so they are not "simplified" away
later.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.part_2_methods.ch04_ddpg import (
    Actor,
    AdaptiveParameterNoise,
    Critic,
    DDPGAgent,
    GaussianNoise,
    ReplayBuffer,
    action_distance,
)

CPU = torch.device("cpu")


def make_agent(**overrides):
    kwargs = dict(
        state_dim=3, action_dim=1, max_action=2.0,
        batch_size=8, buffer_size=500, device=CPU,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    np.random.seed(0)
    return DDPGAgent(**kwargs)


def fill(agent, n=64, rng=None):
    """Pushes n synthetic transitions so ``train`` has something to sample."""
    rng = rng or np.random.default_rng(0)
    for _ in range(n):
        agent.store(
            rng.normal(size=3).astype(np.float32),
            rng.uniform(-2, 2, size=1).astype(np.float32),
            float(rng.normal()),
            rng.normal(size=3).astype(np.float32),
            0.0,
        )


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

def test_actor_respects_the_action_bound():
    actor = Actor(3, 2, max_action=2.0)
    out = actor(torch.randn(32, 3) * 50)  # extreme inputs, to saturate tanh
    assert out.shape == (32, 2)
    assert torch.all(out.abs() <= 2.0 + 1e-6)


def test_actor_output_layer_starts_small():
    """Regression: the module version shipped without ``reset_parameters``.

    Listing 4.2 and the chapter prose both describe the paper's
    U(-3e-3, 3e-3) output init, but ``actor.py`` did not implement it, so the
    book and the companion code taught different networks. Without it a fresh
    actor can emit saturated actions where tanh's gradient is nearly flat.
    """
    actor = Actor(3, 1, max_action=2.0)
    assert actor.l3.weight.abs().max() <= 3e-3
    assert actor.l3.bias.abs().max() <= 3e-3
    # The hidden layers must NOT be shrunk - the paper's fan-in init is
    # nn.Linear's default, and overriding it would break the forward pass scale.
    assert actor.l1.weight.abs().max() > 3e-3


def test_critic_output_layer_starts_small():
    critic = Critic(3, 1)
    assert critic.l3.weight.abs().max() <= 3e-3
    assert critic.l3.bias.abs().max() <= 3e-3


def test_critic_takes_the_action_at_the_second_layer():
    """The paper's architecture choice, asserted on the shapes themselves."""
    critic = Critic(state_dim=3, action_dim=2)
    assert critic.l1.in_features == 3           # state only
    assert critic.l2.in_features == 400 + 2     # state embedding + action
    assert critic(torch.zeros(5, 3), torch.zeros(5, 2)).shape == (5, 1)


def test_critic_gradient_reaches_the_action():
    """The DPG theorem in one assertion: grad_a Q must be non-zero."""
    critic = Critic(3, 1)
    action = torch.zeros(4, 1, requires_grad=True)
    critic(torch.randn(4, 3), action).sum().backward()
    assert action.grad is not None
    assert torch.any(action.grad != 0)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def test_gaussian_noise_is_constant_by_default():
    noise = GaussianNoise(2, sigma=0.2)
    for _ in range(10_000):
        noise.sample()
    assert noise.sigma == pytest.approx(0.2)


def test_gaussian_noise_anneals_linearly_and_then_holds():
    noise = GaussianNoise(1, sigma=0.2, sigma_final=0.05, decay_steps=100)
    assert noise.sigma == pytest.approx(0.2)
    for _ in range(50):
        noise.sample()
    assert noise.sigma == pytest.approx(0.125)   # halfway
    for _ in range(500):
        noise.sample()
    assert noise.sigma == pytest.approx(0.05)    # floors, does not overshoot


def test_gaussian_noise_reset_does_not_restart_annealing():
    """Exploration decays over the run, not within an episode."""
    noise = GaussianNoise(1, sigma=0.2, sigma_final=0.05, decay_steps=100)
    for _ in range(50):
        noise.sample()
    before = noise.sigma
    noise.reset()
    assert noise.sigma == before


def test_gaussian_noise_rejects_an_increasing_schedule():
    with pytest.raises(ValueError):
        GaussianNoise(1, sigma=0.1, sigma_final=0.5)


def test_gaussian_noise_draws_are_independent():
    """The chapter states i.i.d. samples do not alternate in sign.

    Anti-correlated draws would push this well below zero; independent ones
    leave it near it.
    """
    rng = np.random.default_rng(0)
    noise = GaussianNoise(1, sigma=0.2, rng=rng)
    samples = np.array([noise.sample()[0] for _ in range(20_000)])
    correlation = np.corrcoef(samples[:-1], samples[1:])[0, 1]
    assert abs(correlation) < 0.05


def test_parameter_noise_perturbs_a_copy_only():
    actor = Actor(3, 1, 2.0)
    before = [p.detach().clone() for p in actor.parameters()]
    perturbed = AdaptiveParameterNoise(sigma=0.1).perturb(actor)

    assert all(torch.equal(a, b)
               for a, b in zip(before, actor.parameters()))
    assert any(not torch.equal(a, b)
               for a, b in zip(before, perturbed.parameters()))
    assert not any(p.requires_grad for p in perturbed.parameters())


def test_parameter_noise_sigma_tracks_the_target_action_distance():
    torch.manual_seed(0)
    actor = Actor(3, 1, 2.0)
    states = torch.randn(128, 3)
    noise = AdaptiveParameterNoise(sigma=0.001, target_action_stddev=0.2)

    # Starting far below target, sigma must climb.
    for _ in range(300):
        noise.adapt(action_distance(actor, noise.perturb(actor), states))
    assert noise.sigma > 0.001

    distances = [action_distance(actor, noise.perturb(actor), states)
                 for _ in range(20)]
    assert 0.05 < float(np.median(distances)) < 0.8


def test_parameter_noise_rejects_a_non_adapting_coefficient():
    with pytest.raises(ValueError):
        AdaptiveParameterNoise(adaptation_coefficient=1.0)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

def test_replay_buffer_discards_oldest_when_full():
    buf = ReplayBuffer(max_size=5)
    for i in range(8):
        buf.push(np.full(3, i, dtype=np.float32), np.zeros(1), 0.0,
                 np.zeros(3), 0.0)
    assert len(buf) == 5
    kept = sorted(int(t[0][0]) for t in buf.buf)
    assert kept == [3, 4, 5, 6, 7]


def test_replay_buffer_sample_shapes_and_dtype():
    """Regression: rewards and dones must come back as (batch, 1).

    Left as (batch,) they broadcast against the critic's (batch, 1) output into
    a (batch, batch) target, and the Bellman backup is silently wrong rather
    than raising.
    """
    buf = ReplayBuffer(max_size=100)
    for i in range(50):
        buf.push(np.zeros(3), np.zeros(2), float(i), np.zeros(3), 0.0)
    s, a, r, ns, d = buf.sample(16)

    assert s.shape == (16, 3)
    assert a.shape == (16, 2)
    assert r.shape == (16, 1)
    assert ns.shape == (16, 3)
    assert d.shape == (16, 1)
    for tensor in (s, a, r, ns, d):
        assert tensor.dtype == torch.float32


def test_replay_buffer_preserves_transition_contents():
    buf = ReplayBuffer(max_size=10)
    buf.push(np.array([1.0, 2.0, 3.0]), np.array([0.5]), -7.0,
             np.array([4.0, 5.0, 6.0]), 1.0)
    s, a, r, ns, d = buf.sample(1)
    assert s.tolist() == [[1.0, 2.0, 3.0]]
    assert a.tolist() == [[0.5]]
    assert r.tolist() == [[-7.0]]
    assert ns.tolist() == [[4.0, 5.0, 6.0]]
    assert d.tolist() == [[1.0]]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def test_agent_actions_are_bounded_with_and_without_noise():
    agent = make_agent(sigma=5.0)  # noise far larger than the bound
    state = np.zeros(3, dtype=np.float32)
    for explore in (True, False):
        for _ in range(200):
            action = agent.select_action(state, explore=explore)
            assert action.shape == (1,)
            assert np.all(np.abs(action) <= 2.0 + 1e-6)


def test_agent_greedy_action_is_deterministic():
    agent = make_agent()
    state = np.random.default_rng(0).normal(size=3).astype(np.float32)
    first = agent.select_action(state, explore=False)
    for _ in range(10):
        assert np.allclose(agent.select_action(state, explore=False), first)


def test_train_returns_none_until_a_full_batch_exists():
    agent = make_agent(batch_size=32)
    fill(agent, n=31)
    assert agent.train() == (None, None)
    fill(agent, n=1)
    assert agent.train() != (None, None)


def test_train_updates_both_networks():
    agent = make_agent()
    fill(agent)
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    critic_before = [p.detach().clone() for p in agent.critic.parameters()]

    for _ in range(5):
        c_loss, a_loss = agent.train()

    assert np.isfinite(c_loss) and np.isfinite(a_loss)
    assert any(not torch.equal(a, b)
               for a, b in zip(actor_before, agent.actor.parameters()))
    assert any(not torch.equal(a, b)
               for a, b in zip(critic_before, agent.critic.parameters()))


def test_actor_update_restores_critic_gradients():
    """Regression: the critic is frozen during the actor's backward pass.

    That optimisation skips a gradient computation nothing consumes, but if the
    flag is not restored afterwards the critic silently stops learning from
    that point on - a failure that looks like slow convergence, not a crash.
    """
    agent = make_agent()
    fill(agent)
    agent.train()
    assert all(p.requires_grad for p in agent.critic.parameters())

    critic_before = [p.detach().clone() for p in agent.critic.parameters()]
    for _ in range(3):
        agent.train()
    assert any(not torch.equal(a, b)
               for a, b in zip(critic_before, agent.critic.parameters()))


def test_target_networks_are_never_touched_by_the_optimizers():
    agent = make_agent()
    assert not any(p.requires_grad for p in agent.actor_target.parameters())
    assert not any(p.requires_grad for p in agent.critic_target.parameters())

    optimized = {id(p) for group in agent.actor_opt.param_groups
                 for p in group["params"]}
    optimized |= {id(p) for group in agent.critic_opt.param_groups
                  for p in group["params"]}
    assert not any(id(p) in optimized for p in agent.actor_target.parameters())
    assert not any(id(p) in optimized for p in agent.critic_target.parameters())


def test_soft_update_moves_the_target_by_exactly_tau():
    """theta' <- tau * theta + (1 - tau) * theta', checked numerically."""
    agent = make_agent(tau=0.25)
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.fill_(1.0)
        for p in agent.actor_target.parameters():
            p.fill_(0.0)

    agent._soft_update(agent.actor_target, agent.actor)
    for p in agent.actor_target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.25))

    agent._soft_update(agent.actor_target, agent.actor)
    for p in agent.actor_target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.4375))


def test_soft_update_leaves_no_grad_history():
    """Regression: the update used ``.data``, which sidesteps autograd rather
    than satisfying it. ``no_grad`` + ``copy_`` is the supported form."""
    agent = make_agent()
    fill(agent)
    agent.train()
    for p in agent.actor_target.parameters():
        assert p.grad_fn is None
        assert not p.requires_grad


def test_hard_target_update_copies_on_schedule_only():
    agent = make_agent(target_update="hard", hard_update_freq=5)
    fill(agent)

    agent.train()  # step 1 - no copy yet
    assert any(not torch.equal(a, b) for a, b in
               zip(agent.critic.parameters(), agent.critic_target.parameters()))

    for _ in range(4):  # reaches step 5
        agent.train()
    assert all(torch.equal(a, b) for a, b in
               zip(agent.critic.parameters(), agent.critic_target.parameters()))


def test_ablation_without_target_networks_shares_the_online_networks():
    agent = make_agent(use_target_networks=False)
    assert agent.actor_target is agent.actor
    assert agent.critic_target is agent.critic

    fill(agent)
    c_loss, a_loss = agent.train()
    assert np.isfinite(c_loss) and np.isfinite(a_loss)


def test_agent_rejects_an_unknown_target_update_rule():
    with pytest.raises(ValueError):
        make_agent(target_update="periodic")


def test_terminal_transitions_drop_the_bootstrap_term():
    """The (1 - done) mask, verified against a hand-computed target."""
    agent = make_agent(batch_size=1, gamma=0.99)
    state = np.zeros(3, dtype=np.float32)
    agent.store(state, np.zeros(1, dtype=np.float32), 5.0, state, 1.0)

    s, a, r, ns, d = [t.to(CPU) for t in agent.replay.sample(1)]
    with torch.no_grad():
        nq = agent.critic_target(ns, agent.actor_target(ns))
        y = r + agent.gamma * (1.0 - d) * nq
    assert y.item() == pytest.approx(5.0)   # reward only, no bootstrap


def test_save_and_load_round_trip(tmp_path):
    agent = make_agent()
    fill(agent)
    for _ in range(3):
        agent.train()

    path = tmp_path / "ddpg.pt"
    agent.save(path)

    restored = make_agent()
    restored.load(path)

    state = np.random.default_rng(1).normal(size=3).astype(np.float32)
    assert np.allclose(agent.select_action(state, explore=False),
                       restored.select_action(state, explore=False))
    # Targets must be re-synced, not left at their fresh initialization.
    assert all(torch.equal(a, b) for a, b in
               zip(restored.critic.parameters(),
                   restored.critic_target.parameters()))


# ---------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------

def test_training_script_is_reproducible_under_a_seed():
    from src.part_2_methods.ch04_ddpg import train_pendulum

    kwargs = dict(episodes=3, warmup_steps=20, verbose=False)
    first = train_pendulum.main(seed=0, **kwargs)
    second = train_pendulum.main(seed=0, **kwargs)
    assert first == second

    different = train_pendulum.main(seed=1, **kwargs)
    assert different != first


def test_training_script_bootstraps_through_time_limit_truncation():
    """Regression: the loop stored ``float(terminated or truncated)``.

    Pendulum-v1 never terminates - it only truncates at 200 steps - so that
    zeroed the bootstrap on the final transition of every episode, teaching the
    critic that states at t=200 are worthless. Every stored done flag on this
    environment must therefore be 0.0.
    """
    from src.part_2_methods.ch04_ddpg import train_pendulum

    agent_box = {}
    real_agent_cls = train_pendulum.DDPGAgent

    class RecordingAgent(real_agent_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            agent_box["dones"] = []

        def store(self, s, a, r, ns, done):
            agent_box["dones"].append(done)
            super().store(s, a, r, ns, done)

    train_pendulum.DDPGAgent = RecordingAgent
    try:
        train_pendulum.main(seed=0, episodes=2, warmup_steps=10, verbose=False)
    finally:
        train_pendulum.DDPGAgent = real_agent_cls

    dones = agent_box["dones"]
    assert len(dones) == 400          # two full 200-step episodes
    assert set(dones) == {0.0}


@pytest.mark.slow
def test_agent_learns_on_pendulum():
    """The end-to-end claim: DDPG beats a random policy on Pendulum-v1.

    Deliberately a loose bar. The chapter's -200 convergence claim needs a full
    200-episode run; this checks the pipeline learns *something* in a fraction
    of that, which is what catches a wiring bug.
    """
    from src.part_2_methods.ch04_ddpg import train_pendulum

    returns = train_pendulum.main(
        seed=0, episodes=60, warmup_steps=1_000, verbose=False)

    assert len(returns) == 60
    assert np.mean(returns[-10:]) > np.mean(returns[:10])
    assert np.mean(returns[-10:]) > -1_200
