"""Checks fitted Q-evaluation (listing 9.4) against measured returns.

For each trained agent (runs/<name>/checkpoint.pt) this script fits FQE on
the PointMaze dataset with the chapter's own listing, then compares, start
state by start state, FQE's predicted discounted return with the discounted
return the policy actually earns in the environment from that same state.

The environment uses the dataset's own settings (continuing task, a new
random goal each time the goal is reached), so both numbers describe the
same MDP. Rollouts run long enough that gamma**t is negligible at the end.

--single-critic fits and reads q1 alone instead of listing 9.4's minimum
over twin critics, the comparison exercise 9.2 suggests. Its results go to
fqe_vs_true_single.json unless --out says otherwise.

Usage, from the repository root:
    python -m src.part_3_applications.ch09_cql.eval_fqe_vs_true \
        cql sac --fqe-steps 100000 --starts 50
    python -m src.part_3_applications.ch09_cql.eval_fqe_vs_true \
        cql sac --fqe-steps 100000 --starts 50 --single-critic
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.part_3_applications.ch09_cql import cql as L
from src.part_3_applications.ch09_cql import train_cql_vs_sac as T

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

GAMMA = 0.99


def true_returns(env, actor, seeds, horizon):
    starts, returns = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        starts.append(T.obs_vec(obs))
        g, disc = 0.0, 1.0
        for _ in range(horizon):
            s = torch.as_tensor(T.obs_vec(obs), dtype=torch.float32)[None]
            with torch.no_grad():
                a, _ = actor(s, deterministic=True)
            obs, r, term, trunc, _ = env.step(a[0].numpy())
            g += disc * float(r)
            disc *= GAMMA
            if term or trunc:
                break
        returns.append(g)
    return np.array(starts, dtype=np.float32), np.array(returns)


def fqe_single_critic(policy, buffer, fqe_critic, steps, gamma=GAMMA):
    """Listing 9.4 with q1 alone in the target, the loss, and the estimate.

    fqe_critic is still chapter 6's twin Critic, so q1 starts from the same
    weights as in the twin-minimum fit; q2 gets no gradient and never moves.
    """
    target = copy.deepcopy(fqe_critic)
    opt = torch.optim.Adam(fqe_critic.parameters(), lr=3e-4)
    policy.eval()

    for _ in range(steps):
        s, a, r, s2, d = buffer.sample(256)
        with torch.no_grad():
            a2, _ = policy(s2, deterministic=True)
            q1_t, _ = target(s2, a2)
            y = r + gamma * (1 - d) * q1_t
        q1, _ = fqe_critic(s, a)
        loss = F.mse_loss(q1, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        L.soft_update(target, fqe_critic)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--fqe-steps", type=int, default=100_000)
    p.add_argument("--starts", type=int, default=50)
    p.add_argument("--horizon", type=int, default=800)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--single-critic", action="store_true",
                   help="use q1 alone instead of the twin minimum")
    p.add_argument("--out", default=None,
                   help="results file name under runs/ (default "
                        "fqe_vs_true.json, or fqe_vs_true_single.json "
                        "with --single-critic)")
    args = p.parse_args()
    if args.out is None:
        args.out = ("fqe_vs_true_single.json" if args.single_critic
                    else "fqe_vs_true.json")
    torch.set_num_threads(args.threads)

    buffer = L.load_minari(T.DATASET_ID)
    dataset = L.minari.load_dataset(T.DATASET_ID)
    env = dataset.recover_environment(max_episode_steps=args.horizon)
    seeds = np.arange(5000, 5000 + args.starts)
    results = {}

    for name in args.runs:
        torch.manual_seed(0)  # per run, so a run's FQE fit ignores the order
        ck = torch.load(os.path.join(CODE_DIR, "runs", name, "checkpoint.pt"))
        actor = T.Actor(6, 2)
        actor.load_state_dict(ck["actor"])
        actor.eval()

        s0, g_true = true_returns(env, actor, seeds, args.horizon)
        fqe = T.Critic(6, 2)
        if args.single_critic:
            fqe_single_critic(actor, buffer, fqe, args.fqe_steps)
        else:
            L.fitted_q_evaluation(actor, buffer, fqe, steps=args.fqe_steps,
                                  gamma=GAMMA)
        with torch.no_grad():
            st = torch.as_tensor(s0)
            a0, _ = actor(st, deterministic=True)
            q1, q2 = fqe(st, a0)
            est = q1 if args.single_critic else torch.min(q1, q2)
            g_fqe = est.squeeze(-1).numpy()

        corr = (float(np.corrcoef(g_fqe, g_true)[0, 1])
                if np.std(g_true) > 0 else None)
        results[name] = {
            "checkpoint_step": int(ck["step"]),
            "critic": "single" if args.single_critic else "twin_min",
            "fqe_steps": args.fqe_steps,
            "true_mean": float(g_true.mean()), "true_std": float(g_true.std()),
            "fqe_mean": float(g_fqe.mean()), "fqe_std": float(g_fqe.std()),
            "mean_abs_error": float(np.abs(g_fqe - g_true).mean()),
            "per_state_corr": corr,
        }
        print(name, json.dumps(results[name]), flush=True)

    out = os.path.join(CODE_DIR, "runs", args.out)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
