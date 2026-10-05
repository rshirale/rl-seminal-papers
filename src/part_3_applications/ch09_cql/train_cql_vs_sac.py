"""Trains CQL and plain SAC offline on PointMaze UMaze for chapter 9's figure.

Both runs use listings 9.1 and 9.2 unchanged (cql.py) and run listing 9.3's
update inline, so the script can evaluate and checkpoint between steps; the
SAC run is CQL with cql_alpha=0, which is exercise 1. Like listing 9.1, that baseline
keeps the entropy term out of the critic target, so it is SAC without the
entropy backup. Every --eval-every steps the
script logs the mean Q-value the critic assigns to dataset actions, evaluates
the policy in the recovered environment, and saves a checkpoint.

The "plaza" evaluation starts the point in the bottom-left cell and puts the
goal in the top-left cell. A single wall separates them, so the straight-line
route goes through the wall and the real route goes around the U.

Every --log-every steps it also appends to progress.csv the losses, the
measured seconds per training step (evaluation excluded), and, on a fixed
probe of 4,096 dataset transitions, the average dataset Q-value with the
critic loss split into its TD part and its CQL penalty (push_down - push_up,
summed over both critics, before cql_alpha). Those are the curves exercise 6
asks for. Every evaluation also keeps a checkpoint_<step>.pt with the actor
and critic, so the checkpoint at the dataset Q-value's peak can be evaluated
later; checkpoint.pt is always the latest full checkpoint, for --resume.

Usage, from the repository root:
    python -m src.part_3_applications.ch09_cql.train_cql_vs_sac \
        --name cql --cql-alpha 5.0
    python -m src.part_3_applications.ch09_cql.train_cql_vs_sac \
        --name sac --cql-alpha 0.0
    python -m src.part_3_applications.ch09_cql.train_cql_vs_sac \
        --name cql --cql-alpha 5.0 --steps 400000 --resume
"""
import argparse
import copy
import csv
import os
import time

import numpy as np
import torch

from src.part_2_methods.ch06_sac.actor import Actor
from src.part_2_methods.ch06_sac.critic import Critic
from src.part_3_applications.ch09_cql import cql as L

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ID = "D4RL/pointmaze/umaze-v2"
PLAZA_START, PLAZA_GOAL = [3, 1], [1, 1]


def make_eval_env():
    dataset = L.minari.load_dataset(DATASET_ID)
    return dataset.recover_environment(
        continuing_task=False, reset_target=False, max_episode_steps=300)


def obs_vec(obs):
    return np.concatenate([obs["observation"], obs["desired_goal"]])


def rollout(env, actor, seed, options=None):
    obs, _ = env.reset(seed=seed, options=options)
    path, success = [obs["observation"][:2].copy()], False
    goal = obs["desired_goal"].copy()
    for _ in range(300):
        s = torch.as_tensor(obs_vec(obs), dtype=torch.float32)[None]
        with torch.no_grad():
            a, _ = actor(s, deterministic=True)
        obs, _, term, trunc, info = env.step(a[0].numpy())
        path.append(obs["observation"][:2].copy())
        success = success or bool(info.get("success"))
        if term or trunc:
            break
    return np.array(path), goal, success


def evaluate(env, actor, n_random=20):
    plaza = [rollout(env, actor, seed=i,
                     options={"reset_cell": PLAZA_START,
                              "goal_cell": PLAZA_GOAL})
             for i in range(10)]
    rand = [rollout(env, actor, seed=1000 + i) for i in range(n_random)]
    return plaza, rand


def probe_metrics(critic, critic_target, actor, probe):
    """Dataset Q, TD loss, and CQL penalty on the fixed probe batch.

    Calls listing 9.1 twice with identical random draws: cql_alpha=0 gives
    the TD loss alone, cql_alpha=1 adds the penalty once. The forked RNG
    leaves training's random stream, and so --resume, untouched.
    """
    with torch.no_grad(), torch.random.fork_rng():
        dq = torch.min(*critic(probe[0], probe[1])).mean().item()
        torch.manual_seed(0)
        td = L.cql_critic_loss(critic, critic_target, actor, probe,
                               cql_alpha=0.0).item()
        torch.manual_seed(0)
        full = L.cql_critic_loss(critic, critic_target, actor, probe,
                                 cql_alpha=1.0).item()
    return dq, td, full - td


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--cql-alpha", type=float, required=True)
    p.add_argument("--steps", type=int, default=50_000,
                   help="total steps to reach (counting resumed ones)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--log-every", type=int, default=5_000)
    p.add_argument("--eval-every", type=int, default=12_500)
    p.add_argument("--resume", action="store_true",
                   help="continue from runs/<name>/checkpoint.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    out = os.path.join(CODE_DIR, "runs", args.name)
    os.makedirs(out, exist_ok=True)

    buffer = L.load_minari(DATASET_ID)
    env = make_eval_env()
    actor, critic = Actor(6, 2), Critic(6, 2)
    critic_target = copy.deepcopy(critic)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-5)
    probe = buffer.sample(4096)

    ckpt_path = os.path.join(out, "checkpoint.pt")
    log_path = os.path.join(out, "log.csv")
    progress_path = os.path.join(out, "progress.csv")
    start = 0
    if args.resume:
        ck = torch.load(ckpt_path)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        critic_target.load_state_dict(ck["critic_target"])
        critic_opt.load_state_dict(ck["critic_opt"])
        actor_opt.load_state_dict(ck["actor_opt"])
        torch.set_rng_state(ck["torch_rng"])
        start = ck["step"] + 1
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["step", "q_loss", "pi_loss", "dataset_q", "plaza_success",
                 "random_success", "elapsed_s"])
        with open(progress_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "q_loss", "pi_loss",
                                    "sec_per_step", "elapsed_s",
                                    "dataset_q", "td_loss",
                                    "cql_penalty"])

    t0 = time.time()
    train_time = 0.0
    for step in range(start, args.steps + 1):
        t_step = time.time()
        batch = buffer.sample(256)
        q_loss = L.cql_critic_loss(critic, critic_target, actor, batch,
                                   cql_alpha=args.cql_alpha)
        critic_opt.zero_grad()
        q_loss.backward()
        critic_opt.step()

        a_new, log_pi = actor(batch[0])
        q1, q2 = critic(batch[0], a_new)
        pi_loss = (0.2 * log_pi - torch.min(q1, q2)).mean()
        actor_opt.zero_grad()
        pi_loss.backward()
        actor_opt.step()
        L.soft_update(critic_target, critic)
        train_time += time.time() - t_step

        if step > start and step % args.log_every == 0:
            sps = train_time / args.log_every
            row = [step, round(q_loss.item(), 4), round(pi_loss.item(), 4),
                   round(sps, 4), round(time.time() - t0)]
            row += [round(x, 4) for x in
                    probe_metrics(critic, critic_target, actor, probe)]
            with open(progress_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print("progress", " | ".join(str(x) for x in row), flush=True)
            train_time = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                dq = torch.min(*critic(probe[0], probe[1])).mean().item()
            plaza, rand = evaluate(env, actor)
            ps = np.mean([r[2] for r in plaza])
            rs = np.mean([r[2] for r in rand])
            row = [step, q_loss.item(), pi_loss.item(), dq, ps, rs,
                   round(time.time() - t0)]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(" | ".join(str(x) for x in row), flush=True)
            torch.save({"actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "critic_target": critic_target.state_dict(),
                        "critic_opt": critic_opt.state_dict(),
                        "actor_opt": actor_opt.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "step": step}, ckpt_path)
            torch.save({"actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "step": step},
                       os.path.join(out, f"checkpoint_{step:07d}.pt"))
            np.savez(os.path.join(out, f"paths_{step:07d}.npz"),
                     plaza=np.array([r[0] for r in plaza], dtype=object),
                     plaza_goal=plaza[0][1], allow_pickle=True)


if __name__ == "__main__":
    main()
