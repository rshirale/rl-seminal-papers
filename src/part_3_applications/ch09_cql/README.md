# Chapter 9: Offline RL: Learning from static datasets

This directory contains the Python implementations for Chapter 9 of **"Reinforcement Learning: The Foundational Ideas"**. It implements the continuous, SAC-based CQL(H) of Kumar et al. (2020), *Conservative Q-Learning for Offline Reinforcement Learning*, trains it on Minari's `D4RL/pointmaze/umaze-v2` dataset with no environment interaction, and checks the trained policy with fitted Q-evaluation (FQE).

Every algorithm before this chapter learns by acting and watching what happens. Offline RL gets only a fixed log of someone else's actions. Train an off-policy learner such as SAC on that log and its critic overestimates actions the data never covers, because the max in the Bellman target picks exactly those errors and nothing ever corrects them. CQL's answer is to change what the critic learns: push down a soft maximum of the Q-values over all actions, push up the Q-values of the actions actually in the data, and let the actor optimize against the resulting conservative critic.

## File Structure

- `cql.py`: listings 9.1-9.4 exactly as printed. `cql_critic_loss` is the CQL(H) critic loss with the importance-sampled soft maximum over 10 uniform and 10 policy actions per state; `OfflineBuffer` and `load_minari` load a Minari dataset into tensors; `train_offline_cql` is the training loop; `fitted_q_evaluation` estimates a policy's value from the logs alone.
- `train_cql_vs_sac.py`: the runnable experiment. Trains CQL (or plain SAC with `--cql-alpha 0`) on PointMaze, logging losses, seconds per step, and the average dataset Q-value with the TD loss and CQL penalty on a fixed probe batch every `--log-every` steps (`progress.csv`) and, every `--eval-every` steps, success rates on random goals and on the "plaza" task, where the straight line to the goal runs into a wall (`log.csv`). Keeps an actor-and-critic checkpoint per evaluation (`checkpoint_<step>.pt`) for picking the peak-Q checkpoint, plus a full `checkpoint.pt` for `--resume`. Defaults to the chapter's 50,000 steps.
- `eval_fqe_vs_true.py`: fits FQE (listing 9.4) to a trained agent and compares, start state by start state, its predicted discounted return with the return the policy actually earns in the environment. `--single-critic` fits and reads `q1` alone instead of the twin minimum, the comparison exercise 9.2 suggests.
- `Chapter9_CQL.ipynb`: the interactive companion notebook, Colab-ready. It trains SAC and CQL, plots the dataset Q-value, TD loss and CQL penalty, draws the plaza rollouts, and plots FQE against the true return start state by start state. Its code cells *are* chapter 6's `actor.py` and `critic.py`, the four listings in `cql.py`, and the two scripts above, and a test enforces that.
- `__init__.py`: package marker.

## Installation

Chapter 9 needs PyTorch, Minari with its download and storage extras, and Gymnasium Robotics for the PointMaze environment, on Python 3.10 or later. From the project root:
```bash
make install-offline
```
That runs `make install-full` for PyTorch, then installs `requirements-offline.txt`. Without `make`, the equivalent is `pip install torch "minari[hdf5,hf]>=0.5.2" gymnasium-robotics`, plus `"mujoco<=3.10.0"` on Intel macOS, where later MuJoCo releases ship no wheels. The dataset (about 530 MB) downloads on first use and is cached under `~/.minari/datasets`.

## Running the Experiments

From the project root:
```bash
make run-ch9-cql   # CQL at cql_alpha 5.0, 50,000 steps
make run-ch9-sac   # plain SAC on the same logs (cql_alpha 0.0), exercise 9.1
make run-ch9-fqe   # FQE against true returns for both runs, exercise 9.2
```
Each target takes `EXTRA` for further flags, such as `make run-ch9-fqe EXTRA="--single-critic"`. The targets run these commands, which take any `--name` and `--cql-alpha`:
```bash
python -m src.part_3_applications.ch09_cql.train_cql_vs_sac \
    --name cql --cql-alpha 5.0 --steps 50000 --eval-every 12500
python -m src.part_3_applications.ch09_cql.train_cql_vs_sac \
    --name sac --cql-alpha 0.0 --steps 50000 --eval-every 12500
python -m src.part_3_applications.ch09_cql.eval_fqe_vs_true cql sac
```
Results land in `src/part_3_applications/ch09_cql/runs/<name>/`. To continue a run, as exercise 9.6 does from 50,000 to 200,000 steps, repeat its `--name` and `--cql-alpha` and add `--resume --steps 200000`; the script stops if the alpha doesn't match the checkpoint's. On a recent laptop CPU (8 threads) a training step takes about 0.04-0.05 seconds, so 50,000 steps take about 40 minutes; FQE takes a few minutes more.

Interactive notebook: open `Chapter9_CQL.ipynb` locally, or in [Google Colab](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_3_applications/ch09_cql/Chapter9_CQL.ipynb). It runs the same two training runs at the chapter's 50,000 steps, so allow 1.5 to 2 hours on a laptop CPU including FQE. It writes to `runs/` in its working directory, so opened from this folder it shares runs with the scripts and skips any already trained to the requested step count.

The same runs execute on GitHub's machines through `.github/workflows/ch09-runs.yml`, which trains SAC, CQL with alpha 5.0, and CQL with alpha 1.0 on two seeds in parallel and uploads each run's results as an artifact.

## Implementation Notes

- The actor and critic are chapter 6's `Actor` and `Critic`, imported from `src.part_2_methods.ch06_sac`, so CQL is literally SAC with the critic loss swapped, as the paper describes.
- Chapter 6 uses alpha for the SAC entropy temperature and the CQL paper uses alpha for the conservative weight, so the code names them `ent_alpha` and `cql_alpha`.
- The entropy term appears only in the actor loss, not in the critic's Bellman target. That follows the backup in the CQL paper's objective and Algorithm 1, and the one Kumar et al. (2021) assume. On PointMaze's 0/1 reward an entropy term in the target, at a fixed temperature of 0.2, outweighs the reward about a hundredfold once the policy turns nearly deterministic, and drags every Q-value steadily below zero.
- The policy learning rate is 3e-5 against 3e-4 for the critic, the paper's setting.
- `OfflineBuffer` replaces chapter 6's deque-based `ReplayBuffer`, whose random indexing is slow at 1,000,000 transitions.
- The notebook carries the files rather than copies of them. The scripts go through one fixed transformation, defined in `tests/test_ch09_notebook.py`: their `from src.` imports and the `L.`/`T.` prefixes those imports bind are removed, `CODE_DIR` moves to the notebook's setup cell, and the `if __name__ == "__main__":` guard is dropped, because a Jupyter kernel runs every cell as `__main__` and the guarded `main()` would parse the kernel's own command line. The test runs the cells that way, so it catches the guard.
- The average Q-value on dataset actions is logged because Kumar et al. (2021), *A Workflow for Offline Model-Free Robotic RL*, use a dataset Q-value that rises and then falls as their overfitting signal.

## Troubleshooting

- **`TypeError: unsupported operand type(s) for |`** when importing chapter 6: the companion code uses `float | None` type hints, which need Python 3.10 or later.
- **`ImportError: h5py is not installed`**: install the `hdf5` extra, `pip install "minari[hdf5]"`.
- **`Failed to build installable wheels ... mujoco`** on Intel macOS: Gymnasium Robotics pulls in MuJoCo, whose recent releases have no wheels for that platform. `make install-offline` applies the `"mujoco<=3.10.0"` cap; with plain pip, add it to the same command.
- **The notebook parity test fails after I edited a module.** That is the test working. Regenerate the notebook cell from the file rather than editing the cell in place; the failure message names the file.
- **The dataset fails to download**: the `hf` extra provides the Hugging Face client Minari uses for remote datasets.
- **Training stalls for hours on a laptop**: the machine probably slept. `caffeinate -i` prevents only idle sleep; closing the lid on battery still sleeps the Mac.
