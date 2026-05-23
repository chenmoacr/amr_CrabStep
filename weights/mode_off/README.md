---
base_model: google/gemma-4-E2B-it
library_name: peft
tags:
- lora
- transformers
- gemma-4
- crab-step
- amr-CrabStep
---

# Mode B / `mode_off` — Crab Step round 1 (23 neurons suppressed)

LoRA r=8 α=16 adapter trained on 5 (story, Opus-style critique) samples
with **23 amr_wtf inventory neurons forcefully zeroed out** during forward
(see `training_grads.pt["suppress_dict"]` for exact indices).

This is round **1** of the Crab Step iterative SFT loop. See repository
`README.md` for the full method and `outputs/training/summary_mode_off.txt`
for per-step loss curves and the top-30 recruits surfaced by this run.

| field | value |
|---|---|
| host model            | `google/gemma-4-E2B-it` |
| LoRA target modules   | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| LoRA rank / alpha     | 8 / 16 |
| epochs × samples      | 4 × 5 = 20 steps |
| optimizer / LR        | AdamW + cosine, 1e-4 → 1e-6 |
| precision             | BF16 |
| neurons suppressed    | 23 (tiers ∈ {verified, general, lit, guided_diff} ∩ regions ∈ {answer, always}) |

To load:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
import os
base = AutoModelForCausalLM.from_pretrained(os.environ["GEMMA_PATH"],
                                            torch_dtype="bfloat16",
                                            attn_implementation="eager")
model = PeftModel.from_pretrained(base, "weights/mode_off")
```

To reproduce the *training condition* (LoRA + 23-neuron suppression hooks
still installed at inference time), use `core/infer_one_mode.py` with
`CRABSTEP_ADAPTER=weights/mode_off`.
