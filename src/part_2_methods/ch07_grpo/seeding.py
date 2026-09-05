"""Deterministic seeding for the Chapter 7 training script.

Kept deliberately identical in shape to ``ch03_dqn/seeding.py``,
``ch04_ddpg/seeding.py`` and ``ch06_sac/seeding.py``, so the book teaches one
seeding habit rather than four. What differs is which generator matters most:
in the control chapters it was the replay buffer's minibatch draw, and here it
is token sampling. Every completion in every group is drawn from torch's
generator through ``model.generate(do_sample=True)``, so torch's RNG is not a
detail of initialization -- it *is* the exploration.

Unlike chapters 3 to 6, this module is not separately runnable. A seed-variance
demo would mean several full training runs against a downloaded model, which is
sixteen minutes each; ``train_json.py --seed N`` is the same measurement one
seed at a time, and the README says what to compare.
"""

import random

import numpy as np
import torch


def set_seed(seed: int, threads: int = 1) -> None:
    """Seeds every RNG a GRPO run draws from, and pins the thread count.

    Three generators feed a run. ``torch`` seeds the LoRA adapter
    initialization *and* every sampled token -- both the group of G candidates
    and the completions the evaluation draws. ``random`` seeds the dataset
    generator's fallback path and anything a reader adds. ``np.random`` is
    seeded because transformers and peft both reach for it internally, not
    because this chapter's own code does.

    The thread pin is part of seeding rather than a performance tweak, for the
    same reason it is in chapter 6: torch's intra-op parallelism changes the
    order floating-point work is reduced in, so the same seed on an 8-core
    machine and a 4-core one produces different logits, different sampled
    tokens, and a different transcript. On this chapter it bites harder than
    anywhere else in the book, because a single differing token early in a
    completion changes its reward, its advantage, and therefore the update.

    Call this *before* building the model, so adapter initialization is
    covered.

    The pin is a flag here and a constant in chapter 6, and that difference
    is deliberate. Pinning costs chapter 6 nothing measurable -- its networks
    are two-layer MLPs -- but generation on a half-billion parameter model is
    matrix multiplication all the way down, and a single thread is several
    times slower than the machine can go. ``threads=0`` leaves torch's default
    alone and buys that speed back at the price of a reproducible transcript.
    Take it while iterating; put it back before reporting a number.

    None of this makes results identical across platforms, PyTorch versions,
    or dtypes. A run on CUDA in bfloat16 and a run on CPU in float32 will
    diverge on the first sampled token, and that is expected -- see the
    README on what reproduction does and does not cover here.
    """
    if threads:
        torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
