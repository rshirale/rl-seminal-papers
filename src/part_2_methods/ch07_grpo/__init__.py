"""Group Relative Policy Optimization -- Chapter 7 of "RL: The Seminal Papers".

GRPO (Shao et al., 2024) deletes PPO's critic. Instead of learning a value
network to say what a good response would have been worth, it samples a group
of candidates for the same prompt and uses that group's own mean and standard
deviation as the baseline. One trainable model instead of two, which on a 7B
run is roughly 126 GB of VRAM against 238.

Only the dependency-free half of the chapter is re-exported here. ``grpo``
needs torch and ``policy`` needs transformers and peft, and importing either at
package scope would make the reward function -- the part of a GRPO pipeline
readers are most likely to replace with their own -- unimportable without the
whole language-model stack. Chapters 4 to 6 leave their runnable modules out of
``__init__`` for a related reason.

Import the rest by module::

    from src.part_2_methods.ch07_grpo.grpo import grpo_loss
    from src.part_2_methods.ch07_grpo.policy import LoRAPolicy
"""

from .dataset import PROMPT, format_prompt, make_dataset
from .rewards import (SCHEMA_KEYS, SEVERITIES, compute_json_reward,
                      extract_json, is_compliant)

__all__ = [
    "PROMPT",
    "SCHEMA_KEYS",
    "SEVERITIES",
    "compute_json_reward",
    "extract_json",
    "format_prompt",
    "is_compliant",
    "make_dataset",
]
