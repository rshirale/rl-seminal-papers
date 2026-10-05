"""Checks fitted Q-evaluation (listing 9.4) against measured returns.

For each trained agent (runs/<name>/checkpoint.pt) this script fits FQE on
the PointMaze dataset with the chapter's own listing, then compares, start
state by start state, FQE's predicted discounted return with the discounted
return the policy actually earns in the environment from that same state.

The environment uses the dataset's own settings (continuing task, a new
random goal each time the goal is reached), so both numbers describe the
same MDP. Rollouts run long enough that gamma**t is negligible at the end.

Usage, from the repository root:
    python -m src.part_3_applications.ch09_cql.eval_fqe_vs_true \
        cql sac --fqe-steps 100000 --starts 50
"""
import argparse
import json
import os

import numpy as np
import torch

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--fqe-steps", type=int, default=100_000)
    p.add_argument("--starts", type=int, default=50)
    p.add_argument("--horizon", type=int, default=800)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default="fqe_vs_true.json",
                   help="results file name under runs/")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    buffer = L.load_minari(T.DATASET_ID)
    dataset = L.minari.load_dataset(T.DATASET_ID)
    env = dataset.recover_environment(max_episode_steps=args.horizon)
    seeds = np.arange(5000, 5000 + args.starts)
    results = {}

    for name in args.runs:
        ck = torch.load(os.path.join(CODE_DIR, "runs", name, "checkpoint.pt"))
        actor = T.Actor(6, 2)
        actor.load_state_dict(ck["actor"])
        actor.eval()

        s0, g_true = true_returns(env, actor, seeds, args.horizon)
        fqe = T.Critic(6, 2)
        L.fitted_q_evaluation(actor, buffer, fqe, steps=args.fqe_steps)
        with torch.no_grad():
            st = torch.as_tensor(s0)
            a0, _ = actor(st, deterministic=True)
            g_fqe = torch.min(*fqe(st, a0)).squeeze(-1).numpy()

        corr = (float(np.corrcoef(g_fqe, g_true)[0, 1])
                if np.std(g_true) > 0 else None)
        results[name] = {
            "checkpoint_step": int(ck["step"]),
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
