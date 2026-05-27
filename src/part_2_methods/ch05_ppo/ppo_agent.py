import numpy as np
import torch
import torch.nn as nn
from actor_critic import Actor, Critic


class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action,
                 lr=1e-3, gamma=0.9, lam=0.95,
                 eps_clip=0.2, k_epochs=10, batch_size=64):
        self.gamma      = gamma
        self.lam        = lam
        self.eps_clip   = eps_clip
        self.k_epochs   = k_epochs
        self.batch_size = batch_size

        self.actor  = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist  = self.actor(state_t)
            value = self.critic(state_t).item()
        action  = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1).item()
        return action.squeeze(0).numpy(), logprob, value

    def update(self, rollouts):
        states, actions, rewards, next_states, dones, \
            old_logprobs, values = zip(*rollouts)

        states       = torch.FloatTensor(np.array(states))
        actions      = torch.FloatTensor(np.array(actions))
        rewards      = list(rewards)
        dones        = list(dones)
        old_logprobs = torch.FloatTensor(np.array(old_logprobs))
        values       = list(values)

        returns    = []
        advantages = []
        gae        = 0.0
        with torch.no_grad():
            next_value = self.critic(
                torch.FloatTensor(next_states[-1]).unsqueeze(0)
            ).item()

        for i in reversed(range(len(rollouts))):
            next_val = (
                next_value
                if i == len(rollouts) - 1
                else values[i + 1]
            )
            delta = (
                rewards[i]
                + self.gamma * next_val * (1 - dones[i])
                - values[i]
            )
            gae = (
                delta + self.gamma * self.lam * (1 - dones[i]) * gae
            )
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        advantages = torch.FloatTensor(np.array(advantages))
        returns    = torch.FloatTensor(np.array(returns))
        advantages = (
            (advantages - advantages.mean())
            / (advantages.std() + 1e-8)
        )

        n             = len(rollouts)
        approx_kl_sum = 0.0
        clip_frac_sum = 0.0
        update_count  = 0

        for _ in range(self.k_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start: start + self.batch_size]
                mb_states      = states[idx]
                mb_actions     = actions[idx]
                mb_old_lp      = old_logprobs[idx]
                mb_adv         = advantages[idx]
                mb_returns     = returns[idx]

                dist         = self.actor(mb_states)
                logprobs     = dist.log_prob(mb_actions).sum(dim=-1)
                dist_entropy = dist.entropy().sum(dim=-1)
                state_values = self.critic(mb_states).squeeze()

                ratios = torch.exp(logprobs - mb_old_lp)

                surr1 = ratios * mb_adv
                surr2 = torch.clamp(
                    ratios, 1 - self.eps_clip, 1 + self.eps_clip
                ) * mb_adv

                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * nn.MSELoss()(state_values, mb_returns)
                    - 0.01 * dist_entropy
                )

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.critic.parameters()),
                    max_norm=0.5
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl_sum += (
                        (mb_old_lp - logprobs).mean().item()
                    )
                    clip_frac_sum += (
                        ((ratios - 1.0).abs() > self.eps_clip)
                        .float().mean().item()
                    )
                    update_count += 1

        return (
            approx_kl_sum / update_count,
            clip_frac_sum / update_count
        )
