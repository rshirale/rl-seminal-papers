# Chapter 7: Group Relative Policy Optimization (GRPO)

This directory contains the Python implementations for Chapter 7 of **"RL: The Seminal Papers"**. It implements GRPO from Shao et al. (2024), *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, the algorithm that a year later trained DeepSeek-R1. The runnable experiment trains a base language model to emit strict-schema JSON on a laptop CPU, with a rule as its only training signal.

Chapter 5's PPO divided the work between an actor that generates and a critic that estimates what a state is worth. That division is affordable when the actor is a small multilayer perceptron. It stops being affordable when the actor is a language model, because scoring a reasoning trace is about as hard as producing one, so the critic is initialized from a model of the same size — a seven-billion parameter actor implies a seven-billion parameter critic, and fourteen billion parameters resident in VRAM to train seven.

GRPO deletes the critic. Instead of learning a value network to say what a good response would have been worth, it samples a group of *G* candidates for the same prompt and standardizes their rewards against the group's own mean and standard deviation. That pair of statistics is the entire replacement. One trainable model instead of two, roughly 126 GB against 238 on a 7B run, and the cost moves from memory to sampling throughput — which is the trade the algorithm exists to make.

Everything else is chapter 5's, unchanged: the clipped surrogate, the pessimistic minimum, the optimizer, the loop. The one addition is a KL anchor to a frozen reference policy, and in this chapter that reference costs nothing at all, because disabling LoRA adapters turns the same weights back into the original base model.

## File Structure

- `rewards.py`: `extract_json`, `compute_json_reward` and `is_compliant` — listing 7.2, and the entire training signal. Three tiers in order: format, schema conformance, then content scored against *this ticket's* target record. Imports nothing outside the standard library, deliberately: this is the module readers replace with their own.
- `dataset.py`: `make_dataset` and the prompt template. Support-ticket notes generated locally from a seed rather than downloaded, so the task is reproducible, cannot rot, and needs no network.
- `grpo.py`: `group_advantages`, `kl_penalty` and `grpo_loss` — listing 7.4, factored so each term can be tested on its own. Needs torch and nothing else.
- `policy.py`: `LoRAPolicy`. Loads the base model, inserts rank-16 adapters into the attention projections, samples a group as one batch, and scores it under both the current and the reference policy. The only module that needs `transformers` and `peft`.
- `seeding.py`: `set_seed`. Token sampling *is* the exploration here, so torch's generator is on the critical path in a way it never was for the control chapters. The thread pin is a flag rather than a constant — see the Implementation Notes.
- `train_json.py`: the runnable experiment — listing 7.3 wired to a real model. `main()` is importable and returns a `RunResult`; `train_step()` is one GRPO step and can be driven with a stub policy, which is how the tests cover the loop without weights.
- `reward_anatomy.py`: what the reward pays or charges for a range of plausible model outputs — chapter figure 7.6, printed from the reward function's own output rather than illustrated. Instant, and needs nothing installed.
- `group_size.py`: how often a group carries no gradient, by group size — chapter figure 7.7 and exercise 3. Also reads back a real run's per-candidate scores to measure the graded-versus-binary claim.
- `Chapter7_GRPO.ipynb`: the interactive companion notebook (Colab-ready).
- `__init__.py`: exposes the standard-library half of the chapter — the reward function and the dataset. Not `grpo` or `policy`: importing them at package scope would make the reward unimportable without torch and the whole language-model stack.

## Installation

Chapters 1–6 do not need any of this, and neither do two of the three runnable scripts here. Install the stack only for the trainer and the notebook:

```bash
make install-llm
```

That is `torch`, `transformers` and `peft` — roughly half a gigabyte before any weights. The model itself (`Qwen/Qwen2.5-0.5B`, about 1 GB) downloads on the first run and is cached by huggingface afterwards.

`make run-ch7-reward` and `make run-ch7-group-size` need none of it. They run on a bare interpreter, and running them first is the right order: the reward function is what decides whether GRPO is any use on a problem of your own.

## Running the Experiments

Score a range of plausible model outputs with the rule-based reward:

```bash
make run-ch7-reward
```

Instant. Every number it prints is the reward function's own output on the candidate beside it, which is why the tests can pin those eight values against figure 7.6 — edit `rewards.py` and this table moves.

See what group size buys, and what too small a group costs (exercise 3):

```bash
make run-ch7-group-size
```

Also instant, and also not a simulation: for a binary reward the chance that a group of *G* carries no gradient is exactly `p**G + (1-p)**G`, and the table is that expression. Add a real run's scores to get the graded-reward comparison:

```bash
make run-ch7-group-size EXTRA="--history run.json"
```

Train the model:

```bash
make run-ch7-train

# ...or directly, with the chapter's settings spelled out
python -m src.part_2_methods.ch07_grpo.train_json --seed 0 --steps 40 --group-size 8
```

Forty GRPO steps against Qwen2.5-0.5B — a base model that has never been instruction-tuned — with the reward function as the only training signal. No demonstrations, no preference labels, and no critic anywhere in the directory. It is deliberately a miniature of DeepSeek-R1-Zero.

That took 36 minutes on an 8-core Intel CPU at the default pinned single thread, and about a gigabyte of model weights downloads on the first run. `--threads 0` is worth several times the speed if you do not need the transcript to be reproducible, and a free Colab GPU finishes it in a few minutes.

Write the per-step history so `group_size.py` can read it back, and shorten the run while iterating:

```bash
python -m src.part_2_methods.ch07_grpo.train_json --steps 6 --group-size 4 \
    --max-new-tokens 32 --history-out run.json
```

Both analysis targets take `FIGURE_DIR` to write PNG + SVG, like every other figure-producing target in the book:

```bash
make run-ch7-reward     FIGURE_DIR=figures  # ch07-figure-reward-anatomy
make run-ch7-group-size FIGURE_DIR=figures  # ch07-figure-group-size
```

Interactive notebook: open `Chapter7_GRPO.ipynb` locally, or in [Google Colab](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch07_grpo/Chapter7_GRPO.ipynb). A free Colab GPU runs it in a few minutes.

## Implementation Notes

- **The loss is exactly zero on every step, and that is not a bug.** With a single update per exploration stage — the setting the DeepSeek team used, and the one `train_json.py` runs — the behavior policy and the current policy are the same object. The ratio is exactly 1, the clip never binds, and the surrogate collapses to the negative mean of advantages that are standardized within the group and therefore sum to zero by construction. The *gradient* is emphatically not zero: differentiate through the ratio and REINFORCE-with-baseline falls out. This is why the trainer prints the gradient norm rather than the loss, and why `test_grpo.py` asserts both halves of it.

- **The KL term is Schulman's estimator, not a mean log-ratio.** `e^u - u - 1` for `u = log(π_ref/π_θ)`, which is zero when the policies agree and strictly positive otherwise. The naive alternative — averaging the log-ratio — is signed, and when it goes negative the penalty *subtracts* from the loss, rewarding exactly the drift it was added to prevent. The KL term is the only thing holding the policy near a model that still writes readable text, so this is not a small error. It is tested for non-negativity rather than checked by eye.

- **The reference policy is free, and that is the reason LoRA is here.** GRPO's KL term needs log-probabilities under a frozen reference model, which on a full fine-tune means a second copy of the weights — 14 GB for a 7B model. With adapters it costs nothing: `disable_adapter()` bypasses them and the same weights *are* the base model again. The chapter's memory arithmetic is what this directory demonstrates, so paying for a second copy here would undercut the argument.

- **Graduated penalties, not an all-or-nothing zero.** An object that is correct up to a missing closing brace scores −1.00; keys in the wrong case score −0.40; prose that never attempts JSON scores −3.00. Under a rule that zeroed every unparsable output all three would be indistinguishable, and the model would learn nothing from having been one character away. `make run-ch7-reward` prints the spread; it is the learning signal.

  The second benefit is less obvious and is why `group_size.py` exists. A group carries no gradient only when every member scores *identically*. A binary reward needs eight matching verdicts; a graded one needs eight matching numbers out of the range in figure 7.6, which is far rarer. Graduated penalties buy tolerance of small groups.

- **The content tier is scored against this ticket's target, never a constant.** Rewarding a fixed literal would be the easiest term in the function to hack, because the model could satisfy it without reading the input at all. `test_grpo.py` asserts that a memorized record from a different ticket scores below the right one — and note that the memorized record is still strictly *compliant*, because compliance is a structural property. That gap is what the third tier is for.

- **Compliance and mean reward tell different stories, and both are reported.** A reader shown only the first would conclude GRPO taught the model to write JSON; a reader shown only the second would conclude nothing happened. The honest reading is the pass@K / maj@K story in miniature: the capability was already there, and GRPO moved probability mass onto the outputs that were already correct. A jump of this size says the task was nearly solved before training started, and no experiment at this scale can show a half-billion parameter model learning something it could not previously do.

  Reproduced on 2026-09-04 with `make run-ch7-train` at the defaults — seed 0, 40 steps, G = 8, `Qwen/Qwen2.5-0.5B`, float32 on an 8-core Intel CPU with the thread count pinned to 1:

  | | this run | the chapter's |
  | --- | ---: | ---: |
  | Compliance before | 42% | 42% |
  | Compliance after 40 steps | 75% | 83% |
  | Mean group reward, first half | 4.17 | 4.23 |
  | Mean group reward, last half | 4.59 | 4.61 |
  | Steps skipped (zero variance) | 0/40 | 0/40 |
  | Trainable parameters | 2,162,688 of 496,195,456 (0.44%) | ~2.2M of 496M |
  | Wall clock | 36 min | 16 min |

  The compliance figures are 24 samples each, so 75 against 83 is inside the noise of the measurement and should not be read as a difference. What reproduces is the shape: a large move in compliance, a small one in mean reward, and no step skipped. The wall clock does not reproduce and is not supposed to — this run pinned one thread on an Intel CPU.

- **CI runs this chapter with none of its dependencies installed, and that is the design working.** `.github/workflows/tests.yml` runs the full suite on every push and pull request, across Python 3.10, 3.12 and 3.13, and it does not install `requirements-llm.txt`. `LoRAPolicy.load` is covered by faking the three constructors it calls, the objective needs only torch, and the reward and the dataset need nothing at all — so the one test that genuinely wants weights, the notebook's top-to-bottom run, skips itself unless they are already in the huggingface cache. The dependency split described above is what makes that possible.

  What it costs is worth stating plainly. CI reports 319 passed and 1 skipped where a machine with the weights cached reports 320, and the skipped one is the only test that proves the notebook a reader opens actually runs. A green check does not cover it. To cover it yourself, run `make run-ch7-train` once to populate the cache, then `make test-all` ([example run](https://github.com/rshirale/rl-seminal-papers/actions/runs/33985565975)).

- **The thread pin is a flag here and a constant in chapter 6.** Both chapters pin `torch.set_num_threads(1)` as part of seeding, because intra-op parallelism changes the order floating-point work is reduced in and therefore, here, which token gets sampled. The difference is what it costs: chapter 6's two-layer MLPs do not notice, while generation from a half-billion parameter model is matrix multiplication all the way down. `--threads 0` leaves torch's default alone and runs several times faster without a reproducible transcript. Take it while iterating; put it back before reporting a number.

- **MPS is opt-in on Apple Silicon, and that is measured rather than assumed.** On a 0.5B model in float32, MPS produced 4.2 tokens/second against 7.4 on plain CPU, because several generation ops fall back to the CPU anyway and each fallback pays a device transfer. `--force-mps` is there for readers who want to check on their own hardware.

- **The base model, not the `-Instruct` variant.** Qwen2.5-0.5B-Instruct already solves this task — measured at 20/20 perfect — which would leave GRPO nothing to learn and turn the run into a demonstration that instruction tuning works. Starting from a base model with a rule as the only signal is what makes this a miniature of R1-Zero. A test asserts the notebook has not drifted to the instruct variant.

- **Log-probabilities are averaged over tokens, not summed.** That is GRPO's documented short-response bias: each token of a long completion contributes less to the loss than each token of a short one, so the model learns to reach the answer in as few tokens as possible. It is the right bias for a one-line JSON object and the wrong one for long chain-of-thought, which is exactly what DAPO's token-level aggregation corrects. Sum instead of mean if your task rewards length.

- **Sampling is batched, and that is what makes G affordable.** GRPO's cost against PPO is that it needs *G* complete generations per prompt instead of one. Generating them as a single padded batch shares the matrix multiplications; on a GPU the extra rows are close to free. Left padding is required for this — right-padded rows would be continued from their padding rather than from the prompt, which produces fluent nonsense and no error at all.

- **The evaluation prompts are also training prompts, by default.** The chapter's run measures compliance on the first six dataset rows, and the loop trains on those rows too. Completions are sampled fresh each time, so the number measures the policy rather than recall of a stored string — but it is not a generalization check, and it should not be read as one. `--eval-holdout` scores the tail of the dataset instead, which a 40-step run over 64 rows never reaches. Use it when the question is whether the behavior transferred rather than whether it appeared.

- **The zero-variance guard never fired, and that is measurable rather than lucky.** A group whose members all score alike carries no signal, so `train_step` skips it rather than taking a null update — the degenerate condition DAPO's dynamic sampling was designed to remove. Not one step in 40 was skipped, in the chapter's run or in the reproduction above, and `group_size.py --history` says why. Pooling the run's 320 candidate scores:

  | | G = 2 | G = 4 | G = 8 | G = 16 |
  | --- | ---: | ---: | ---: | ---: |
  | Graded reward, measured | 18.5% | 1.2% | 0.0% | 0.0% |
  | Binary reward at the same success rate | 50.7% | 13.6% | 1.1% | 0.0% |

  Same run and same command as the table above, read back with `make run-ch7-group-size EXTRA="--history run.json"`. The reward emitted 19 distinct scores across those 320 candidates, and a group collapses only when every member lands on one value — one of 19 here, against one of two under a binary rule. That is the chapter's claim about graduated penalties buying tolerance of small groups, measured on a real run rather than argued. Note the caveat the script prints: candidates within one group are drawn for the same prompt and so are more alike than these pooled draws, which makes the graded row a floor.

## Troubleshooting

- **`ImportError: Chapter 7's trainer needs transformers and peft.`** Run `make install-llm`. The reward and group-size targets do not need it and will keep working without it — that split is deliberate.

- **The first run stalls for several minutes with no output.** It is downloading about a gigabyte of model weights. Subsequent runs read them from the huggingface cache. `HF_HUB_OFFLINE=1` forces the cache and fails loudly instead of reaching for the network.

- **My numbers do not match the ones above.** They will not match line for line, and should be read as a shape rather than a checksum. Sampling temperature, thread count, dtype and PyTorch version all change which token is drawn, and a single differing token early in a completion changes its reward, its advantage, and the update. What should transfer: compliance rises substantially, mean group reward rises slightly, and the gap between those two facts is the result.

  Compliance also carries real sampling noise — 6 rows at 4 samples each is 24 draws — so read a difference of 40 points as real and a difference of 4 as nothing.

- **The run is far too slow.** Two things to try, in this order: `--threads 0`, which unpins the thread count and is worth several times the speed on a multicore CPU, and a shorter budget while iterating (`--steps 6 --group-size 4 --max-new-tokens 32`). A free Colab GPU finishes the full run in a few minutes; the notebook is the same experiment.

- **Every step reports `skipped (zero variance)`.** The group's members are all scoring identically. On a short debug run with `--max-new-tokens` set very low this is expected — the model cannot finish an object in the budget, so every candidate scores the same failure. Raise the token budget before suspecting the algorithm.

- **The loss prints as `0.000` and I cannot tell if anything is training.** Watch the gradient norm column instead; see the first implementation note. If the gradient norm is also zero, the group had no variance and the step was skipped.

- **`ModuleNotFoundError: No module named 'src'`.** Run the `-m` form from the project root, not from this directory. The scripts also work when run directly (`python train_json.py`) from inside this directory, which is the fallback the `if __package__` branches exist for — but the two are not interchangeable from the same working directory.

- **The notebook's install line fails on Colab.** The specifiers must stay quoted — an unquoted `>=` is read by the shell as a redirection. The setup cell already quotes them; keep them that way if you edit it.
