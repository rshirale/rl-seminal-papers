"""Train a base LLM to emit strict-schema JSON with GRPO, and nothing else.

This is listing 7.3 wired to a real model. It is deliberately a miniature of
DeepSeek-R1-Zero: the starting point is Qwen2.5-0.5B, a base model that has
never been instruction-tuned, and the only training signal is the rule-based
reward in ``rewards.py``. There are no demonstrations, no preference labels,
and no critic anywhere in this file.

Usage::

    python -m src.part_2_methods.ch07_grpo.train_json
    python -m src.part_2_methods.ch07_grpo.train_json --steps 40 --group-size 8

    # ...or directly, from this directory
    python train_json.py --seed 0

Needs ``transformers`` and ``peft`` (``make install-llm``) and about a gigabyte
of model weights, downloaded once and cached by huggingface. The default 40
steps took 36 minutes on an 8-core Intel CPU with the thread count pinned to 1,
against the 16 minutes the chapter reports; ``--threads 0`` unpins it and is
worth several times the speed, at the cost of a reproducible transcript.

What to expect, and what not to. Compliance -- the strict pass/fail of
``is_compliant`` -- moves a lot: the chapter's run went from 42% to 83%. Mean
group reward moves much less, 4.23 to 4.61, and that gap is the more honest
picture of what happened. The reward was already close to its ceiling, and most
of the gain came from converting near-misses into exact hits. Read it as the
pass@K / maj@K story in miniature: the capability was already there, and GRPO
moved probability mass onto the outputs that were already correct. A jump from
42 to 83% says the task was nearly solved before training started; no
experiment at this scale can show a half-billion parameter model learning
something it could not previously do.
"""

import argparse
import json
import time
from collections import namedtuple

import torch

if __package__:
    from .dataset import SEED, format_prompt, make_dataset
    from .grpo import CLIP_EPS, KL_BETA, grpo_loss
    from .policy import (MAX_NEW_TOKENS, MODEL_ID, TEMPERATURE, LoRAPolicy,
                         pick_device)
    from .rewards import compute_json_reward
    from .seeding import set_seed
else:  # pragma: no cover - only used by direct script execution.
    from dataset import SEED, format_prompt, make_dataset
    from grpo import CLIP_EPS, KL_BETA, grpo_loss
    from policy import (MAX_NEW_TOKENS, MODEL_ID, TEMPERATURE, LoRAPolicy,
                        pick_device)
    from rewards import compute_json_reward
    from seeding import set_seed

#: The chapter's configuration. G = 8 sits inside the 4-to-16 range that
#: open-source reimplementations converged on, well below DeepSeekMath's 64 --
#: a compute tradeoff rather than a disagreement about the algorithm.
STEPS = 40
GROUP_SIZE = 8
LEARNING_RATE = 1e-5

#: Gradient-norm clip. Not part of GRPO; ordinary hygiene for a run whose
#: advantage scale is set by a group of eight samples and can spike when one
#: member of a group scores far from the rest.
MAX_GRAD_NORM = 1.0

#: Evaluation budget: the first six dataset rows, four samples each. Small
#: because it is paid twice (before and after) and every sample is a full
#: generation. See ``LoRAPolicy.compliance`` on how much noise 24 samples
#: carries -- enough that a 4-point move means nothing.
EVAL_ROWS = 6
EVAL_SAMPLES = 4

#: Below this, a group's members have all scored alike and the advantages are
#: all zero, so the step would be a null update. Compared against the standard
#: deviation of the group's rewards.
ZERO_VARIANCE_TOL = 1e-6

#: What one run reports back. ``history`` is one dict per step; ``baseline``
#: and ``after`` are strict compliance rates around the run.
RunResult = namedtuple("RunResult", "history baseline after elapsed skipped")


def train_step(policy, row, group_size, optimizer, clip_eps=CLIP_EPS,
               kl_beta=KL_BETA, max_grad_norm=MAX_GRAD_NORM, **sample_kwargs):
    """One GRPO step on one prompt. Returns the record appended to history.

    The four numbered stages are Algorithm 1 of the DeepSeekMath paper, and
    they are the whole method::

        1. sample a group of G candidates from the current policy
        2. score every candidate with the rule
        3. take log-probs under the current policy and the frozen reference
        4. apply the objective and update

    ``old_logp`` is ``logp.detach()``, which deserves a word. We take a single
    update per exploration stage -- the setting the DeepSeek team used -- so
    the behavior policy that drew the samples *is* the current policy, and the
    ratio is exactly 1. Keeping the term rather than dropping it is what lets a
    reader raise the number of updates per group without rewriting the
    objective, and it is why the loss prints as 0.000 while the gradient is
    large. This is the single most confusing thing in the run; see ``grpo.py``.
    """
    prompt = format_prompt(row)

    # 1. Sample a group of candidates from the current policy.
    texts, prompt_ids, gen_ids = policy.sample_group(prompt, group_size,
                                                     **sample_kwargs)

    # 2. Score every candidate with the rule-based reward.
    scores = [compute_json_reward(t, row["target"]) for t in texts]
    rewards = torch.tensor(scores, dtype=torch.float32, device=policy.device)

    # A group whose members all score the same carries no signal at all: every
    # advantage in it is zero. Skip it rather than take a null step. This is
    # the degenerate condition DAPO's dynamic sampling was designed to remove,
    # and on the chapter's 40-step run it never fired once -- because the
    # reward is graded rather than binary, so eight completions have to land on
    # exactly the same score before the variance vanishes.
    if rewards.std() < ZERO_VARIANCE_TOL:
        return {"reward": rewards.mean().item(), "rewards": scores,
                "loss": None, "gnorm": None, "skipped": True}

    # 3. Log-probs under the current policy and the frozen reference.
    logp = policy.sequence_logprob(prompt_ids, gen_ids, use_adapter=True)
    ref_logp = policy.sequence_logprob(prompt_ids, gen_ids, use_adapter=False)
    old_logp = logp.detach()

    # 4. GRPO objective and update.
    loss = grpo_loss(logp, old_logp, ref_logp, rewards, group_size,
                     clip_eps=clip_eps, kl_beta=kl_beta)
    optimizer.zero_grad()
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(),
                                           max_grad_norm)
    optimizer.step()

    # Every candidate's score is kept, not just the group mean. ``group_size.py
    # --history`` reads them back to compute how often a group of any G would
    # have collapsed on this run's own measured score distribution, which is
    # the graded-versus-binary comparison the chapter makes in prose.
    return {"reward": rewards.mean().item(), "rewards": scores,
            "loss": loss.item(), "gnorm": float(gnorm), "skipped": False}


def main(seed=SEED, steps=STEPS, group_size=GROUP_SIZE, lr=LEARNING_RATE,
         model_id=MODEL_ID, device=None, force_mps=False,
         eval_rows=EVAL_ROWS, eval_samples=EVAL_SAMPLES, eval_holdout=False,
         max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
         kl_beta=KL_BETA, clip_eps=CLIP_EPS, dataset_size=64,
         threads=1, print_every=5, verbose=True, history_out=None):
    """Runs the whole experiment and returns a :class:`RunResult`.

    Importable and parameterized for the same reason every trainer in this
    book is: nothing here should be reachable only from the command line.
    ``group_size.py`` reads the history this writes, and ``train_step`` below
    is separable enough that the tests drive a full GRPO step with a stub
    policy -- no weights, no download, no network.
    """
    set_seed(seed, threads=threads)
    device = device or pick_device(force_mps=force_mps)

    if verbose:
        print(f"device: {device}\nmodel : {model_id}")

    policy = LoRAPolicy.load(model_id=model_id, device=device)
    if verbose:
        trainable, total = policy.parameter_counts()
        print(f"trainable: {trainable:,} of {total:,} parameters "
              f"({100 * trainable / total:.2f}%)")

    data = make_dataset(dataset_size)
    # The chapter's run evaluates on the first rows, which the loop also
    # trains on -- the completions are sampled fresh, so it measures the
    # policy rather than recall, but it is not a generalization check.
    # `eval_holdout` evaluates on the tail instead, which a 40-step run over a
    # 64-row dataset never reaches. Use it when the question is whether the
    # behavior transferred rather than whether it appeared.
    eval_slice = data[-eval_rows:] if eval_holdout else data[:eval_rows]

    # Measure before training, so the number after training means something.
    baseline = policy.compliance(eval_slice, n_each=eval_samples,
                                 max_new_tokens=max_new_tokens,
                                 temperature=temperature)
    if verbose:
        print(f"baseline compliance: {baseline:.0%}\n")

    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=lr)

    history = []
    started = time.time()
    for step in range(steps):
        record = train_step(
            policy, data[step % len(data)], group_size, optimizer,
            clip_eps=clip_eps, kl_beta=kl_beta,
            max_new_tokens=max_new_tokens, temperature=temperature)
        record["step"] = step
        history.append(record)

        if verbose and (step % print_every == 0 or step == steps - 1):
            elapsed = time.time() - started
            if record["skipped"]:
                print(f"step {step:3d}  reward {record['reward']:6.2f}  "
                      f"skipped (zero variance)  ({elapsed:.0f}s)")
            else:
                print(f"step {step:3d}  reward {record['reward']:6.2f}  "
                      f"grad {record['gnorm']:6.3f}  ({elapsed:.0f}s)")

    elapsed = time.time() - started
    after = policy.compliance(eval_slice, n_each=eval_samples,
                              max_new_tokens=max_new_tokens,
                              temperature=temperature)
    skipped = sum(r["skipped"] for r in history)

    result = RunResult(history=history, baseline=baseline, after=after,
                       elapsed=elapsed, skipped=skipped)
    # Written before the summary is printed. A run of this length is expensive
    # enough that losing it to a formatting bug in the reporting code would be
    # a genuine setback -- which is exactly how the guard in `report` for a
    # one-step run came to be written.
    if history_out:
        write_history(result, history_out, seed=seed, steps=steps,
                      group_size=group_size, lr=lr, model_id=model_id,
                      threads=threads)
    if verbose:
        report(result, steps, printer=print)
    return result


def report(result, steps, printer=print):
    """Prints the two numbers that matter, and the caveat that goes with them.

    Compliance and mean reward are reported side by side on purpose. A reader
    shown only the first would conclude GRPO taught the model to write JSON; a
    reader shown only the second would conclude nothing happened. Both are
    true, and the gap between them is the result.
    """
    scored = [r["reward"] for r in result.history if not r["skipped"]]

    after_label = f"after {steps} steps"
    printer(f"\ndone in {result.elapsed:.0f}s")
    printer(f"{'baseline compliance':<20}: {result.baseline:.0%}")
    printer(f"{after_label:<20}: {result.after:.0%}")
    printer(f"{'change':<20}: {result.after - result.baseline:+.0%}")

    # Halves need at least two scored steps to exist. A one-step run is a
    # smoke test, not a measurement, so it just reports the single mean.
    if len(scored) >= 2:
        half = len(scored) // 2
        first = sum(scored[:half]) / half
        last = sum(scored[half:]) / len(scored[half:])
        printer(f"\nmean group reward, first half: {first:.2f}")
        printer(f"mean group reward, last half : {last:.2f}")
    elif scored:
        printer(f"\nmean group reward: {scored[0]:.2f} (one scored step)")
    printer(f"steps skipped (zero variance): "
            f"{result.skipped}/{len(result.history)}")


def write_history(result, path, **config):
    """Writes the run to JSON, for ``group_size.py --history`` to read.

    The config keys travel with the numbers deliberately: a history file with
    no record of its group size or seed cannot be read against anything, and
    the chapter READMEs' provenance rule applies to files as much as to
    tables.
    """
    payload = {
        "config": config,
        "baseline": result.baseline,
        "after": result.after,
        "elapsed": result.elapsed,
        "skipped": result.skipped,
        "history": result.history,
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5-0.5B to emit strict-schema JSON with GRPO.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=STEPS,
                        help="GRPO steps; one prompt and one group each.")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE,
                        help="G -- completions sampled per prompt. Below 4 "
                             "the baseline collapses; see group_size.py.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--kl-beta", type=float, default=KL_BETA,
                        help="KL coefficient. Its effect is not monotonic.")
    parser.add_argument("--model", dest="model_id", default=MODEL_ID)
    parser.add_argument("--eval-rows", type=int, default=EVAL_ROWS)
    parser.add_argument("--eval-holdout", action="store_true",
                        help="Evaluate on dataset rows the run never trains "
                             "on, rather than on the first rows as the "
                             "chapter's run does.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--threads", type=int, default=1,
                        help="Torch thread count. 1 pins it, which is part of "
                             "seeding; 0 leaves torch's default and runs "
                             "several times faster without a reproducible "
                             "transcript.")
    parser.add_argument("--force-mps", action="store_true",
                        help="Use MPS on Apple Silicon. Measured slower than "
                             "CPU for this model size; see policy.py.")
    parser.add_argument("--history-out", metavar="PATH", default=None,
                        help="Write the per-step history as JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(seed=args.seed, steps=args.steps, group_size=args.group_size,
         lr=args.lr, kl_beta=args.kl_beta, model_id=args.model_id,
         eval_rows=args.eval_rows, eval_holdout=args.eval_holdout,
         max_new_tokens=args.max_new_tokens,
         temperature=args.temperature, force_mps=args.force_mps,
         threads=args.threads, history_out=args.history_out)
