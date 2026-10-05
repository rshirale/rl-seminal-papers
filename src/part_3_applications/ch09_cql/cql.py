"""Listings 9.1-9.4 of chapter 9: the continuous CQL critic loss, the offline
buffer and training loop, and fitted Q-evaluation, exactly as printed."""
# --- Listing 9.1 ---------------------------------------------------------
import math
import torch
import torch.nn.functional as F

def soft_lse(q_rand, q_pi, log_pi, log_unif):
    cat = torch.cat([q_rand - log_unif, q_pi - log_pi], 1)
    return torch.logsumexp(cat, dim=1).mean()

def cql_critic_loss(critic, critic_target, actor, batch,
                    gamma=0.99, cql_alpha=5.0, n=10):
    s, a, r, s2, d = batch
    b, act_dim = a.shape

    with torch.no_grad():
        a2, _ = actor(s2)
        q1_t, q2_t = critic_target(s2, a2)
        y = r + gamma * (1 - d) * torch.min(q1_t, q2_t)

    q1, q2 = critic(s, a)
    td_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)  # <1>

    s_rep = s.repeat_interleave(n, dim=0)
    with torch.no_grad():
        a_rand = torch.rand(b * n, act_dim,
                            device=s.device) * 2 - 1
        a_pi, logp_pi = actor(s_rep)
    log_unif = act_dim * math.log(0.5)
    logp_pi = logp_pi.view(b, n)

    q1_r, q2_r = critic(s_rep, a_rand)
    q1_p, q2_p = critic(s_rep, a_pi)
    push_down = (                                    # <2>
        soft_lse(q1_r.view(b, n), q1_p.view(b, n),
                 logp_pi, log_unif)
        + soft_lse(q2_r.view(b, n), q2_p.view(b, n),
                   logp_pi, log_unif))
    push_up = q1.mean() + q2.mean()                  # <3>

    return td_loss + cql_alpha * (push_down - push_up)  # <4>


# --- Listing 9.2 ---------------------------------------------------------
import minari
import numpy as np

class OfflineBuffer:
    def __init__(self, *columns):
        self.cols = [torch.as_tensor(np.concatenate(c),
                                     dtype=torch.float32)
                     for c in columns]

    def sample(self, batch_size):
        idx = torch.randint(len(self.cols[0]), (batch_size,))
        return [c[idx] for c in self.cols]

def load_minari(dataset_id="D4RL/pointmaze/umaze-v2"):
    dataset = minari.load_dataset(dataset_id,
                                  download=True)     # <1>
    s, a, r, s2, d, starts = [], [], [], [], [], []
    for ep in dataset.iterate_episodes():
        obs = np.concatenate(                        # <2>
            [ep.observations["observation"],
             ep.observations["desired_goal"]], axis=1)
        starts.append(obs[:1])
        s.append(obs[:-1])
        s2.append(obs[1:])
        a.append(ep.actions)
        r.append(ep.rewards[:, None])
        d.append(ep.terminations[:, None])
    buffer = OfflineBuffer(s, a, r, s2, d)
    buffer.starts = OfflineBuffer(starts).cols[0]
    return buffer

# --- Listing 9.3 ---------------------------------------------------------
import copy

def soft_update(target, source, tau=0.005):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.lerp_(sp.data, tau)

def train_offline_cql(actor, critic, buffer, steps=50_000,
                      batch_size=256, cql_alpha=5.0,
                      ent_alpha=0.2):
    critic_target = copy.deepcopy(critic)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    actor_opt = torch.optim.Adam(actor.parameters(),
                                 lr=3e-5)            # <3>

    for step in range(steps):
        batch = buffer.sample(batch_size)
        q_loss = cql_critic_loss(critic, critic_target, actor,
                                 batch, cql_alpha=cql_alpha)
        critic_opt.zero_grad()
        q_loss.backward()
        critic_opt.step()

        s = batch[0]
        a_new, log_pi = actor(s)
        q1, q2 = critic(s, a_new)
        pi_loss = (ent_alpha * log_pi
                   - torch.min(q1, q2)).mean()      # <4>
        actor_opt.zero_grad()
        pi_loss.backward()
        actor_opt.step()

        soft_update(critic_target, critic)
        if step % 10_000 == 0:
            print(f"step {step:>7} | Q loss {q_loss.item():8.2f}"
                  f" | pi loss {pi_loss.item():8.2f}")


# --- Listing 9.4 ---------------------------------------------------------
def fitted_q_evaluation(policy, buffer, fqe_critic,
                        steps=100_000, gamma=0.99):
    target = copy.deepcopy(fqe_critic)
    opt = torch.optim.Adam(fqe_critic.parameters(), lr=3e-4)
    policy.eval()

    for _ in range(steps):
        s, a, r, s2, d = buffer.sample(256)
        with torch.no_grad():
            a2, _ = policy(s2, deterministic=True)   # <1>
            q1_t, q2_t = target(s2, a2)
            y = r + gamma * (1 - d) * torch.min(q1_t, q2_t)
        q1, q2 = fqe_critic(s, a)                    # <2>
        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        soft_update(target, fqe_critic)

    with torch.no_grad():                            # <3>
        s0 = buffer.starts
        a0, _ = policy(s0, deterministic=True)
        return torch.min(*fqe_critic(s0, a0)).mean()
