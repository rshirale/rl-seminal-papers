"""The policy: a base model with LoRA adapters, and its free reference copy.

This is the one module in the chapter that needs the language-model stack
(``transformers`` and ``peft``). Install it with ``make install-llm``; nothing
else in the directory imports it, so the reward function and the objective stay
runnable without it.

Two decisions here carry the chapter's argument, and both are worth reading
before the code:

**The reference policy is free.** GRPO's KL term needs log-probabilities under
a frozen reference model. On a full fine-tune that means a second copy of the
weights resident in memory -- 14 GB for a 7B model at two bytes a parameter.
With LoRA it costs nothing at all: ``disable_adapter()`` bypasses the adapters
and the *same* weights are the original base model again. That is not a trick
specific to this chapter; it is why LoRA and GRPO pair as well as they do.

**The base model, not the instruct variant.** ``Qwen2.5-0.5B`` has never been
instruction-tuned. The ``-Instruct`` variant already solves this task, which
would leave GRPO nothing to learn and produce a chapter-shaped result with no
content in it. Starting from a base model with a rule as the only signal is
what makes the run a miniature of DeepSeek-R1-Zero rather than a demonstration
that instruction tuning works.
"""

import os

import torch

#: A base model, deliberately not the ``-Instruct`` variant. See the module
#: docstring. At 0.5B it downloads in about a gigabyte and trains on a laptop.
MODEL_ID = "Qwen/Qwen2.5-0.5B"

#: LoRA rank and scaling. Rank 16 on the four attention projections leaves
#: roughly 2.2M of the model's 496M parameters trainable -- under half a
#: percent, which is the chapter's memory argument made concrete.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

#: Sampling settings for the group. Temperature above 1.0 would widen the
#: group's spread and with it the advantage signal, at the cost of sampling
#: text the policy is unlikely to produce again; 0.9 with nucleus sampling is
#: the usual compromise and is what the chapter's run used.
MAX_NEW_TOKENS = 72
TEMPERATURE = 0.9
TOP_P = 0.95


def pick_device(force_mps=False):
    """CUDA if present, else CPU -- and MPS only when explicitly asked for.

    The exception is measured, not superstition. On an Apple Silicon Mac with
    a model this small in float32, MPS produced 4.2 tokens/second against 7.4
    on plain CPU, because several generation ops fall back to the CPU anyway
    and each fallback pays a device transfer. MPS wins on larger models; it
    loses here, so it is opt-in.
    """
    if torch.cuda.is_available():
        return "cuda"
    if force_mps and torch.backends.mps.is_available():
        # Several ops used during generation have no MPS kernel. Without this
        # the run dies partway through the first group rather than at load.
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return "mps"
    return "cpu"


class LoRAPolicy:
    """A LoRA-adapted causal LM that can sample groups and score them.

    Wraps the tokenizer and the peft model together because every operation
    the trainer needs -- sampling, scoring, evaluating -- requires both, and
    passing them around as a pair is how the notebook's globals became a
    module. The trainer talks only to this object.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_id=MODEL_ID, device=None, rank=LORA_RANK,
             lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES,
             dtype=torch.float32):
        """Downloads (or reads from cache) the base model and adapts it.

        Imported lazily rather than at module scope so that ``import
        policy`` fails with this function's error message rather than an
        ImportError from three frames down, and so the rest of the chapter
        can be imported without the LLM stack installed.
        """
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - install-path message.
            raise ImportError(
                "Chapter 7's trainer needs transformers and peft. Install "
                "them with `make install-llm` from the project root."
            ) from exc

        device = device or pick_device()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Left padding, not the default right. Generation continues from the
        # last position of the batch, so right-padded rows would be continued
        # from their padding rather than from the prompt -- which produces
        # fluent nonsense and no error at all.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(model_id,
                                                    torch_dtype=dtype)
        config = LoraConfig(
            r=rank, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
            target_modules=list(target_modules), task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, config).to(device)
        return cls(model, tokenizer, device)

    def trainable_parameters(self):
        """The adapter parameters -- what the optimizer is given."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def parameter_counts(self):
        """``(trainable, total)``, the numbers behind the "half a percent"."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        return trainable, total

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_group(self, prompt_text, n, max_new_tokens=MAX_NEW_TOKENS,
                     temperature=TEMPERATURE, top_p=TOP_P):
        """Samples ``n`` completions for one prompt, as a single batch.

        Batching is what makes GRPO affordable. The algorithm's cost against
        PPO is that it needs G complete generations per prompt instead of one,
        and generating them as one padded batch costs far less than G separate
        calls -- the forward passes share the same matrix multiplications, and
        on a GPU the extra rows are close to free.

        Returns ``(texts, prompt_ids, gen_ids)``. The two tensors are handed
        straight to :meth:`sequence_logprob`; re-tokenizing the decoded text
        instead would be subtly wrong, because decode-then-encode is not the
        identity for every token sequence.
        """
        encoded = self.tokenizer([prompt_text] * n, return_tensors="pt",
                                 padding=True).to(self.device)
        out = self.model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = out[:, encoded["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return texts, encoded["input_ids"], gen_ids

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def sequence_logprob(self, prompt_ids, gen_ids, use_adapter=True):
        """Mean per-token log-probability of ``gen_ids``, one value per row.

        ``use_adapter=False`` scores under the frozen reference policy by
        disabling the adapters -- the free reference model of the module
        docstring. That branch runs under ``no_grad``, because nothing about
        the reference is being trained and holding its graph would double the
        memory for no reason.

        The logits are shifted by one before gathering: position ``t``
        predicts token ``t+1``, so the distribution over the first generated
        token is the one at the prompt's last position. Off-by-one here is
        silent -- the run trains, the numbers look plausible, and every
        log-probability is attributed to the wrong token.

        **Mean, not sum.** Dividing by the token count is what the chapter's
        note on the short-response bias is about: averaging within a sample
        means each token of a long completion contributes less to the loss
        than each token of a short one, which teaches the model to answer in
        as few tokens as possible. That is fine here -- the target output is a
        one-line JSON object and brevity is a virtue -- and it is exactly the
        bias DAPO's token-level aggregation corrects for long chain-of-thought.
        Sum instead of mean if your task rewards length.
        """
        full = torch.cat([prompt_ids, gen_ids], dim=1)
        # Explicit rather than ambient, and covering the whole computation
        # rather than the forward pass alone: the sampling that produced these
        # ids runs under no_grad, and a caller who scores inside that block
        # would otherwise get a detached tensor and a loss that trains
        # nothing. The gather and the mean are as much a part of the graph as
        # the model call is, so narrowing this to the forward would build a
        # graph and then drop it -- the cost of the guard without the guard.
        grad_mode = torch.enable_grad() if use_adapter else torch.no_grad()
        with grad_mode:
            if use_adapter:
                logits = self.model(full).logits
            else:
                with self.model.disable_adapter():
                    logits = self.model(full).logits

            logits = logits[:, prompt_ids.shape[1] - 1:-1, :]
            logp = torch.log_softmax(logits.float(), dim=-1)
            token_logp = logp.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)

            # Padding is not part of the policy's output, so it is not scored.
            # Note that pad_token is eos_token on this model, so the
            # terminating EOS is masked out along with the padding after it --
            # a simplification that costs one token's log-probability per
            # sequence and keeps the mask a one-liner.
            mask = (gen_ids != self.tokenizer.pad_token_id).float()
            return ((token_logp * mask).sum(dim=1)
                    / mask.sum(dim=1).clamp(min=1))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def compliance(self, rows, n_each=4, **sample_kwargs):
        """Fraction of sampled completions that are strictly schema-compliant.

        Sampled rather than greedy, and that is the honest choice: the policy
        being trained is a distribution, and a greedy decode would report the
        behavior of a mode the deployment never uses. It also means this
        number carries sampling noise -- with 6 rows at 4 samples each it moves
        by a few points between calls on an unchanged model, so read a
        difference of 40 points as real and a difference of 4 as nothing.
        """
        if __package__:
            from .dataset import format_prompt
            from .rewards import is_compliant
        else:  # pragma: no cover - only used by direct script execution.
            from dataset import format_prompt
            from rewards import is_compliant

        hits = total = 0
        for row in rows:
            texts, _, _ = self.sample_group(format_prompt(row), n_each,
                                            **sample_kwargs)
            for text in texts:
                hits += bool(is_compliant(text, row["target"]))
                total += 1
        return hits / total if total else 0.0
